import pandas as pd

from ematching.skills import build_lexicon, extract_skills, normalize_text


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
