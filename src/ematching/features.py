"""Critères (variables) calculés pour une paire profil-offre.

Chaque critère vaut entre 0 et 1 et se lit simplement, pour pouvoir expliquer le
score à l'utilisateur. Ne sont JAMAIS utilisés : le métier source
(source_occupation_uri, desired_job, target_job), qui sert à calculer les labels
provisoires (fuite de données), ni le nom, l'âge, le genre ou la photo.
"""

import math

SEP = "|"
EDUCATION_RANK = {"Bac": 1, "Bac+2": 2, "Bac+3": 3, "Bac+5": 5}
UNKNOWN = 0.5  # valeur neutre quand l'information manque (ni bonus, ni pénalité)
FORBIDDEN = ("source_occupation_uri", "desired_job", "target_job")


def split_skills(value) -> list[str]:
    """« a|b|c » -> ["a", "b", "c"] (liste vide si la valeur manque)."""
    if isinstance(value, (list, set, tuple)):
        return list(value)
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(SEP) if item.strip()]


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def coverage(owned: list[str], wanted: list[str]) -> float:
    """Part des compétences voulues que le profil possède (libellés identiques)."""
    if not wanted:
        return 1.0  # l'offre ne demande rien de particulier
    return len(set(owned) & set(wanted)) / len(set(wanted))


def experience_score(years, min_years) -> float:
    """1 si l'expérience suffit ; sinon la part atteinte (2 ans sur 4 -> 0,5)."""
    if _missing(years):
        return UNKNOWN
    if _missing(min_years) or min_years <= 0:
        return 1.0
    return min(float(years) / float(min_years), 1.0)


def education_score(profile_level, offer_level) -> float:
    """1 si le diplôme est suffisant, 0,5 s'il manque un niveau, 0 sinon."""
    have, need = EDUCATION_RANK.get(profile_level), EDUCATION_RANK.get(offer_level)
    if have is None or need is None:
        return UNKNOWN
    if have >= need:
        return 1.0
    return 0.5 if need - have <= 1 else 0.0


def location_score(profile_city, offer_city) -> float:
    """1 si même ville, 0 sinon (distance réelle : amélioration possible)."""
    if _missing(profile_city) or _missing(offer_city):
        return UNKNOWN
    return 1.0 if profile_city == offer_city else 0.0


def pair_features(profile: dict, offer: dict) -> dict[str, float]:
    """Critères « règles » d'une paire (sans le sens : voir matching/semantic.py)."""
    owned = split_skills(profile.get("skills"))
    return {
        "skills_exact": coverage(owned, split_skills(offer.get("required_skills"))),
        "nice_to_have": coverage(owned, split_skills(offer.get("nice_to_have_skills"))),
        "experience": experience_score(
            profile.get("years_experience"), offer.get("min_experience_years")
        ),
        "education": education_score(
            profile.get("education_level"), offer.get("education_level")
        ),
        "location": location_score(profile.get("city"), offer.get("city")),
    }
