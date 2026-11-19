import numpy as np
import pandas as pd
import pytest

from ematching.data import PROCESSED_DIR, PROJECT_ROOT
from ematching.skills import (
    SkillExtractor,
    choose_threshold,
    build_lexicon,
    extract_skills,
    label_variants,
    normalize_text,
)


def test_normalize_text_keeps_inner_dots_only():
    assert normalize_text("Node.js, Python.") == "node.js python"
    assert normalize_text("Déboguer un logiciel...") == "deboguer un logiciel"


def test_extract_skills_finds_known_labels():
    skills = pd.DataFrame(
        {
            "conceptUri": ["u1", "u2"],
            "preferredLabel": ["Déboguer un logiciel", "Gérer les budgets"],
            "altLabels": ["corriger un programme", None],
        }
    )
    lexicon = build_lexicon(skills)
    assert extract_skills("Il sait déboguer un logiciel.", lexicon) == {"u1"}
    assert extract_skills("Il sait corriger un programme", lexicon) == {"u1"}
    assert extract_skills("Aucune compétence ici", lexicon) == set()


# --- SkillExtractor (extraction améliorée) -------------------------------------

REFERENTIAL = pd.DataFrame(
    {
        "skillUri": ["py", "sql1", "sql2", "stock", "r", "web"],
        "preferredLabel": [
            "Python (programmation informatique)",
            "SQL",
            "SQL",
            "gérer des stocks",
            "R",
            "programmation web",
        ],
        "altLabels": [None, None, None, None, None, None],
    }
)
SYNONYMS = [("développement web", "programmation web")]


@pytest.fixture
def extractor():
    return SkillExtractor(REFERENTIAL, SYNONYMS)


def found(extractor, text):
    return {match.uri: match.method for match in extractor.extract(text)}


def test_label_variants_drop_parenthesis():
    assert "python" in label_variants("Python (programmation informatique)")


def test_exact_match_keeps_every_uri_of_a_shared_label(extractor):
    assert found(extractor, "Python, SQL") == {
        "py": "exact",
        "sql1": "exact",
        "sql2": "exact",
    }


def test_short_acronyms_are_ignored(extractor):
    assert "r" not in found(extractor, "Habilitation B1V / BR, langage R")


def test_noun_form_and_typos_are_recognised(extractor):
    assert found(extractor, "Gestion des stocks") == {"stock": "forme"}
    assert found(extractor, "gestion des stoks") == {"stock": "approché"}


def test_synonyms_and_no_false_fuzzy_match(extractor):
    assert found(extractor, "Développement web") == {"web": "synonyme"}
    assert found(extractor, "Développement de portefeuille clients") == {}


def test_match_keeps_its_source_line(extractor):
    match = extractor.extract("Profil\nRequêtes SQL sur PostgreSQL")[0]
    assert match.source == "Requêtes SQL sur PostgreSQL"
    assert match.confidence == 1.0


def test_project_synonyms_point_to_existing_esco_labels():
    # Lève une erreur si un synonyme vise un libellé absent du référentiel.
    SkillExtractor.from_files(
        PROCESSED_DIR / "skills_referential.csv",
        PROJECT_ROOT / "data" / "lexicon" / "skills_synonyms.csv",
    )


# --- Méthode sémantique (avec un faux modèle : rien n'est téléchargé) ---------


class FakeEncoder:
    """Faux modèle d'embeddings : vecteurs fixés à la main, normalisés."""

    VECTORS = {
        "gérer des stocks": [1.0, 0.0, 0.0],
        "Encadrement des magasiniers": [0.96, 0.28, 0.0],  # proche : cos = 0,96
        "Suivi des réapprovisionnements": [0.9, 0.0, 0.44],  # assez proche : 0,9
    }

    def __init__(self):
        self.calls = 0

    def _encode(self, texts):
        self.calls += 1
        rows = [self.VECTORS.get(text, [0.0, 0.0, 1.0]) for text in texts]
        rows = np.array(rows, dtype=float)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    encode_queries = encode_documents = _encode


def semantic_extractor(threshold=0.95, scope="unmatched"):
    return SkillExtractor(
        REFERENTIAL,
        SYNONYMS,
        encoder=FakeEncoder(),
        accept_threshold=threshold,
        semantic_scope=scope,
    )


def test_semantic_match_above_threshold_is_accepted():
    match = semantic_extractor().extract("Encadrement des magasiniers")[0]
    assert (match.uri, match.method, match.status) == ("stock", "sémantique", "ok")
    assert match.confidence <= 0.8  # toujours sous les méthodes lexicales


def test_semantic_match_just_below_threshold_is_flagged_for_review():
    extractor = semantic_extractor(threshold=0.92)
    match = extractor.extract("Suivi des réapprovisionnements")[0]
    assert match.status == "à revoir"
    assert (
        semantic_extractor(threshold=0.99).extract("Suivi des réapprovisionnements")
        == []
    )


def test_lexical_methods_keep_priority_over_semantic():
    match = semantic_extractor(threshold=0.5).extract("Gestion des stocks")[0]
    assert match.method == "forme"


def test_semantic_is_optional_and_needs_a_threshold():
    assert SkillExtractor(REFERENTIAL).encoder is None
    with pytest.raises(ValueError):
        SkillExtractor(REFERENTIAL, encoder=FakeEncoder())


def test_line_vectors_are_cached():
    extractor = semantic_extractor()
    extractor.extract("Encadrement des magasiniers")
    calls = extractor.encoder.calls
    extractor.accept_threshold = 0.9  # régler le seuil ne ré-encode rien
    extractor.extract("Encadrement des magasiniers")
    assert extractor.encoder.calls == calls


def test_choose_threshold_guarantees_precision():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, False]
    # Seuil 0,7 : 3 réponses acceptées, toutes justes (100 %) ; 0,6 : 3/4 = 75 %.
    assert choose_threshold(scores, correct, 0.8, min_accepted=1) == 0.7
    assert choose_threshold(scores, correct, 0.7, min_accepted=1) == 0.6
    assert choose_threshold(scores, correct, 0.8, min_accepted=10) is None


def test_no_fuzzy_match_on_single_words():
    referential = pd.DataFrame(
        {"skillUri": ["auto"], "preferredLabel": ["automatisme"], "altLabels": [None]}
    )
    # « automatisé » n'est pas une faute de frappe pour « automatisme ».
    assert SkillExtractor(referential).extract("Reporting automatisé") == []
