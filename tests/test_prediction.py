from collections import Counter

import numpy as np
import pandas as pd
import pytest

from ematching.data import PROCESSED_DIR, PROJECT_ROOT
from ematching.matching.semantic import CONFIG
from ematching.prediction import Recommender, default_weights_path

CV_DIR = PROJECT_ROOT / "data" / "samples" / "cv"
# Métier des offres (pour vérifier les résultats ; jamais donné au moteur).
JOB_OF = dict(
    pd.read_csv(PROCESSED_DIR / "offers.csv")[["offer_id", "target_job"]].values
)


@pytest.fixture(scope="module")
def recommender():
    return Recommender(model=None)  # sans embeddings : rapide, rien à télécharger


def top_jobs(result):
    return Counter(JOB_OF[r.offer_id] for r in result.results)


@pytest.mark.parametrize(
    "cv_file, job",
    [
        ("cv_01_developpeur_junior.pdf", "developpeur"),
        ("cv_03_comptable.pdf", "comptable"),
        ("cv_06_charge_rh.docx", "rh"),
    ],
)
def test_cv_gets_offers_of_its_job(recommender, cv_file, job):
    result = recommender.recommend(CV_DIR / cv_file)
    assert len(result.results) == 5
    assert top_jobs(result)[job] >= 4


def test_result_is_explained(recommender):
    result = recommender.recommend(CV_DIR / "cv_03_comptable.pdf")
    best = result.results[0]
    assert result.profile["education_level"] == "Bac+3"
    assert all(
        m.method in {"exact", "synonyme", "forme", "approché"} for m in result.skills
    )
    assert best.common_skills and set(best.parts) >= {"skills", "experience"}
    assert result.offers[0]["offer_id"] == best.offer_id
    assert "target_job" not in result.offers[0]  # le métier source n'est pas exposé


def test_corrected_profile_and_warnings(recommender):
    result = recommender.recommend_text("Bonjour, je cherche un emploi.")
    assert "Aucune compétence reconnue : compléter la liste." in result.warnings
    corrected = {**result.profile, "skills": "comptabilité|écritures comptables"}
    assert top_jobs(recommender.recommend_profile(corrected))["comptable"] >= 3


class CountingEncoder:
    """Faux modèle qui compte les textes encodés."""

    spec = {"skill_threshold": 0.99}

    def __init__(self):
        self.encoded = 0

    def _encode(self, texts):
        self.encoded += len(texts)
        rng = np.random.default_rng(0)
        rows = rng.normal(size=(len(texts), 8))
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    encode_queries = encode_documents = _encode


def test_offers_are_encoded_once():
    encoder = CountingEncoder()
    recommender = Recommender(encoder=encoder)
    after_warm_up = encoder.encoded
    recommender.recommend(CV_DIR / "cv_03_comptable.pdf")
    recommender.recommend(CV_DIR / "cv_03_comptable.pdf")
    # Après le préchauffage, seules les compétences du CV sont encodées (une fois).
    assert encoder.encoded - after_warm_up <= len(
        recommender.recommend(CV_DIR / "cv_03_comptable.pdf").skills
    )


def test_default_weights_follow_the_model():
    assert default_weights_path(None) == CONFIG  # sans modèle : a priori
    assert default_weights_path("none") == CONFIG
    assert default_weights_path("modele-inconnu") == CONFIG  # pas de fichier appris
    learned = PROJECT_ROOT / "config" / "matching_learned_qwen3-0.6b.toml"
    if learned.exists():
        assert default_weights_path("qwen3-0.6b") == learned
