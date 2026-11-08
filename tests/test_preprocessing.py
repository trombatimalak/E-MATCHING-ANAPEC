from ematching.preprocessing import dedupe_skills, split_skills


def test_split_skills_ignores_empty_values():
    assert split_skills("a|b||c") == ["a", "b", "c"]
    assert split_skills(float("nan")) == []


def test_dedupe_skills_ignores_case_and_accents():
    assert dedupe_skills(["Gérer", "gerer", "Budget"]) == ["Gérer", "Budget"]
