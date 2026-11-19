import numpy as np
import pytest

from ematching.features import (
    education_score,
    experience_score,
    location_score,
    pair_features,
)
from ematching.matching.semantic import SemanticEngine, load_weights

PROFILE = {
    "profile_id": "P1",
    "skills": "gérer des stocks|utiliser un logiciel de tableur",
    "years_experience": 2,
    "education_level": "Bac+2",
    "city": "Agadir",
}
OFFER = {
    "offer_id": "O1",
    "required_skills": "gérer des stocks|opérations d'entreposage",
    "nice_to_have_skills": "utiliser un logiciel de tableur",
    "min_experience_years": 4,
    "education_level": "Bac+3",
    "city": "Agadir",
}


# --- Critères (features.py) ------------------------------------------------------


def test_rule_criteria():
    assert experience_score(2, 4) == 0.5
    assert experience_score(5, 0) == 1.0
    assert education_score("Bac+2", "Bac+3") == 0.5  # un niveau en dessous
    assert education_score("Bac", "Bac+5") == 0.0
    assert location_score("Agadir", "Rabat") == 0.0
    assert education_score(None, "Bac+3") == 0.5  # inconnu : valeur neutre


def test_pair_features():
    assert pair_features(PROFILE, OFFER) == {
        "skills_exact": 0.5,
        "nice_to_have": 1.0,
        "experience": 0.5,
        "education": 0.5,
        "location": 1.0,
    }


def test_forbidden_columns_do_not_change_the_score():
    engine = SemanticEngine()
    leaky = {**PROFILE, "desired_job": "logistique", "source_occupation_uri": "x"}
    assert engine.score(leaky, OFFER).score == engine.score(PROFILE, OFFER).score


# --- Moteur sans embeddings --------------------------------------------------------


def test_engine_without_encoder_explains_its_score():
    result = SemanticEngine().score(PROFILE, OFFER)
    assert result.common_skills == ["gérer des stocks"]
    assert result.missing_skills == ["opérations d'entreposage"]
    assert "semantic" not in result.parts  # critère ignoré sans modèle
    weights = load_weights()
    used = {k: w for k, w in weights.items() if k != "semantic"}
    expected = sum(w * result.parts[k] for k, w in used.items()) / sum(used.values())
    assert result.score == pytest.approx(expected, abs=1e-4)


def test_weights_are_normalised():
    assert sum(load_weights().values()) == pytest.approx(1.0)


def test_rank_returns_best_offers_first():
    other = {**OFFER, "offer_id": "O2", "required_skills": "souder des métaux"}
    ranked = SemanticEngine().rank(PROFILE, [other, OFFER], k=1)
    assert [r.offer_id for r in ranked] == ["O1"]


# --- Moteur avec un faux modèle d'embeddings ---------------------------------------


class FakeEncoder:
    """« opérations d'entreposage » est proche de « utiliser un logiciel de
    tableur » (cos = 0,95) ; le reste est orthogonal."""

    VECTORS = {
        "gérer des stocks": [1, 0, 0],
        "utiliser un logiciel de tableur": [0, 1, 0],
        "opérations d'entreposage": [0, 0.95, 0.312],
    }

    def _encode(self, texts):
        rows = np.array([self.VECTORS.get(t, [0, 0, 1]) for t in texts], dtype=float)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    encode_queries = encode_documents = _encode


def test_equivalent_skill_counts_as_common_with_explanation():
    engine = SemanticEngine(FakeEncoder(), skill_threshold=0.9)
    result = engine.score(PROFILE, OFFER)
    assert result.common_skills == [
        "gérer des stocks",
        "opérations d'entreposage (≈ utiliser un logiciel de tableur)",
    ]
    assert result.missing_skills == []
    assert result.parts["skills"] == 1.0
    assert 0 <= result.parts["semantic"] <= 1


def test_threshold_is_required_for_unknown_models():
    with pytest.raises(ValueError):
        SemanticEngine(FakeEncoder())
