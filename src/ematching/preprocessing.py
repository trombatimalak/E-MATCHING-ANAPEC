"""Nettoyage et contrôles de qualité des données."""

import pandas as pd

from ematching.skills import normalize_text

SEP = "|"


def split_skills(value) -> list[str]:
    """Transforme 'a|b|c' en liste, en ignorant les valeurs vides."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split(SEP) if item.strip()]


def dedupe_skills(skills: list[str]) -> list[str]:
    """Supprime les doublons (après normalisation) en gardant l'ordre."""
    seen, result = set(), []
    for skill in skills:
        key = normalize_text(skill)
        if key and key not in seen:
            seen.add(key)
            result.append(skill)
    return result


def clean_skill_column(series: pd.Series) -> pd.Series:
    """Nettoie une colonne de compétences séparées par '|'."""
    return series.map(lambda value: SEP.join(dedupe_skills(split_skills(value))))


def clean_dataframe(
    data: pd.DataFrame, id_col: str, skill_cols: list[str]
) -> pd.DataFrame:
    """Supprime les identifiants en double et nettoie les compétences."""
    data = data.drop_duplicates(subset=id_col).copy()
    for column in skill_cols:
        data[column] = clean_skill_column(data[column])
    return data


def find_issues(
    profiles: pd.DataFrame,
    offers: pd.DataFrame,
    pairs: pd.DataFrame,
    test_grid: pd.DataFrame,
) -> list[str]:
    """Retourne la liste des problèmes détectés (liste vide si tout va bien)."""
    issues = []

    if not profiles["profile_id"].is_unique:
        issues.append("profile_id non uniques")
    if not offers["offer_id"].is_unique:
        issues.append("offer_id non uniques")

    for name, data, columns in [
        ("profiles", profiles, ["skills", "cv_text", "desired_job"]),
        ("offers", offers, ["required_skills", "description", "title"]),
    ]:
        for column in columns:
            missing = data[column].isna().sum()
            if missing:
                issues.append(f"{name}.{column} : {missing} valeurs manquantes")

    if (profiles["years_experience"] < 0).any():
        issues.append("expérience négative dans profiles")
    if (offers["min_experience_years"] < 0).any():
        issues.append("expérience négative dans offers")
    if not profiles["synthetic"].all() or not offers["synthetic"].all():
        issues.append("des lignes ne sont pas marquées synthetic")

    for name, data, column in [
        ("profiles", profiles, "skills"),
        ("offers", offers, "required_skills"),
    ]:
        for value in data[column].dropna():
            items = split_skills(value)
            if len(items) != len(dedupe_skills(items)):
                issues.append(f"doublons de compétences dans {name}")
                break

    unknown_profiles = set(pairs["profile_id"]) - set(profiles["profile_id"])
    unknown_offers = set(pairs["offer_id"]) - set(offers["offer_id"])
    if unknown_profiles or unknown_offers:
        issues.append("paires avec des identifiants inconnus")

    train, test = pairs[pairs["split"] == "train"], pairs[pairs["split"] == "test"]
    if set(train["profile_id"]) & set(test["profile_id"]):
        issues.append("FUITE : profils présents dans train et test")
    if set(train["offer_id"]) & set(test["offer_id"]):
        issues.append("FUITE : offres présentes dans train et test")

    expected = test["profile_id"].nunique() * test["offer_id"].nunique()
    if len(test_grid) != expected:
        issues.append(f"grille de test : {len(test_grid)} lignes au lieu de {expected}")

    return issues
