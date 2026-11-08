"""Génère des profils et des offres synthétiques à partir d'ESCO."""

import random

import pandas as pd

from ematching.data import PROCESSED_DIR, load_esco
from ematching.skills import normalize_text

SEED = 42
N_OFFERS = 150
N_PROFILES = 300
CITIES = ["Casablanca", "Rabat", "Marrakech", "Tanger", "Fès", "Agadir", "Oujda"]
CONTRACTS = ["CDI", "CDD", "Stage", "Intérim"]
LANGUAGES = ["Français", "Français|Anglais", "Arabe|Français", "Arabe|Français|Anglais"]
SECTORS = {
    "developpeur": "Informatique",
    "comptable": "Finance et comptabilité",
    "commercial": "Commerce et vente",
    "maintenance": "Industrie",
    "rh": "Ressources humaines",
    "logistique": "Logistique et transport",
}
SEP = "|"
JUNIOR_WORDS = ("assistant", "agent", "technicien")
SENIOR_WORDS = ("responsable", "ingenieur", "expert", "auditeur", "chef")


def short_label(label: str) -> str:
    """Garde la forme masculine d'un libellé ESCO (avant le '/')."""
    return label.split("/")[0].strip()


def education_options(title: str) -> list[str]:
    """Niveaux d'études plausibles selon le niveau du poste."""
    key = normalize_text(title)
    if any(word in key for word in JUNIOR_WORDS):
        return ["Bac", "Bac+2", "Bac+3"]
    if any(word in key for word in SENIOR_WORDS):
        return ["Bac+3", "Bac+5"]
    return ["Bac+2", "Bac+3", "Bac+5"]


def unique(items: list) -> list:
    """Supprime les doublons en conservant l'ordre."""
    return list(dict.fromkeys(items))


def skills_by_occupation(
    target: pd.DataFrame, relations: pd.DataFrame, labels: dict
) -> dict:
    """Retourne {uri: (compétences essentielles, optionnelles)} en français."""
    rel = relations[relations["occupationUri"].isin(target["conceptUri"])].copy()
    rel["label"] = rel["skillUri"].map(labels)
    rel = rel.dropna(subset=["label"])
    result = {}
    for uri, group in rel.groupby("occupationUri"):
        is_essential = group["relationType"] == "essential"
        result[uri] = (
            group.loc[is_essential, "label"].tolist(),
            group.loc[~is_essential, "label"].tolist(),
        )
    return result


def sample(items: list, k: int, rng: random.Random) -> list:
    return rng.sample(items, min(k, len(items)))


def by_job(occupations: list) -> dict:
    """Regroupe les métiers ESCO par métier cible."""
    groups: dict = {}
    for occ in occupations:
        groups.setdefault(occ["target_job"], []).append(occ)
    return groups


def make_offers(occupations: list, skills: dict, rng: random.Random) -> pd.DataFrame:
    groups = by_job(occupations)
    jobs = list(groups)
    rows = []
    for i in range(N_OFFERS):
        job = jobs[i % len(jobs)]
        pool = groups[job]
        occ = pool[(i // len(jobs)) % len(pool)]
        essential, optional = skills[occ["conceptUri"]]
        title = short_label(occ["preferredLabel"])
        required = sample(essential, 6, rng)
        nice = sample([s for s in optional if s not in required], 3, rng)
        contract = rng.choice(CONTRACTS)
        rows.append(
            {
                "offer_id": f"O{i + 1:04d}",
                "title": title,
                "target_job": occ["target_job"],
                "sector": SECTORS[occ["target_job"]],
                "city": rng.choice(CITIES),
                "contract_type": contract,
                "education_level": rng.choice(education_options(title)),
                "min_experience_years": (
                    0 if contract == "Stage" else rng.choice([0, 1, 2, 3, 5])
                ),
                "required_skills": SEP.join(required),
                "nice_to_have_skills": SEP.join(nice),
                "description": (
                    f"Nous recherchons un(e) {title}. "
                    f"Compétences requises : {', '.join(required)}."
                ),
                "source_occupation_uri": occ["conceptUri"],
                "synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def make_profiles(occupations: list, skills: dict, rng: random.Random) -> pd.DataFrame:
    groups = by_job(occupations)
    jobs = list(groups)
    rows = []
    for i in range(N_PROFILES):
        occ = rng.choice(groups[jobs[i % len(jobs)]])
        other = rng.choice(occupations)
        essential, optional = skills[occ["conceptUri"]]
        noise = skills[other["conceptUri"]][0]
        own = sample(essential, rng.randint(3, 8), rng)
        own += sample(optional, rng.randint(0, 3), rng)
        own += sample(noise, rng.randint(0, 2), rng)
        own = unique(own)
        years = rng.choice([0, 1, 2, 3, 4, 6, 8])
        title = short_label(occ["preferredLabel"])
        education = rng.choice(education_options(title))
        rows.append(
            {
                "profile_id": f"P{i + 1:04d}",
                "desired_job": occ["target_job"],
                "education_level": education,
                "years_experience": years,
                "city": rng.choice(CITIES),
                "languages": rng.choice(LANGUAGES),
                "skills": SEP.join(own),
                "cv_text": (
                    f"Profil {title}. Formation : {education}. "
                    f"{years} an(s) d'expérience. Compétences : {', '.join(own)}."
                ),
                "source_occupation_uri": occ["conceptUri"],
                "synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(SEED)
    target = pd.read_csv(PROCESSED_DIR / "target_occupations.csv")
    relations = load_esco(
        "occupationSkillRelations",
        usecols=["occupationUri", "relationType", "skillUri"],
    )
    skills_fr = load_esco("skills", usecols=["conceptUri", "preferredLabel"])
    labels = dict(zip(skills_fr["conceptUri"], skills_fr["preferredLabel"]))

    skills = skills_by_occupation(target, relations, labels)
    occupations = [
        occ for occ in target.to_dict("records") if occ["conceptUri"] in skills
    ]

    offers = make_offers(occupations, skills, rng)
    profiles = make_profiles(occupations, skills, rng)
    offers.to_csv(PROCESSED_DIR / "offers.csv", index=False)
    profiles.to_csv(PROCESSED_DIR / "profiles.csv", index=False)

    print(offers["target_job"].value_counts().to_string())
    print(profiles["desired_job"].value_counts().to_string())
    print(f"\n{len(offers)} offres, {len(profiles)} profils (synthétiques)")


if __name__ == "__main__":
    main()
