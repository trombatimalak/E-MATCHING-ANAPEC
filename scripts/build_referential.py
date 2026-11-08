"""Construit le référentiel de compétences pour les métiers cibles."""

import pandas as pd

from ematching.data import PROCESSED_DIR, load_esco
from ematching.skills import normalize_text

# Mots-clés (sans accents) cherchés dans les libellés de métiers ESCO.
TARGET_JOBS = {
    "developpeur": ["developpeur"],
    "comptable": ["comptable"],
    "commercial": ["commercial"],
    "maintenance": ["maintenance"],
    "rh": ["ressources humaines", "recrutement"],
    "logistique": ["logisticien", "logistique"],
}
EXCLUDE = [
    "directeur",
    "directrice",
    "aviation",
    "aeronautique",
    "aeroportuaire",
    "pilote",
    "marine",
    "eolien",
    "papier",
    "ingenieur technico commercial",
    "automobile",
    "incendie",
    "responsable recrutement",
]
MAX_PER_JOB = 5


def select_occupations(occupations: pd.DataFrame, relations: pd.DataFrame):
    """Retient, par métier cible, les 5 métiers ESCO les mieux documentés."""
    counts = relations.groupby("occupationUri").size()
    occupations = occupations.assign(
        norm=occupations["preferredLabel"].map(normalize_text),
        n_skills=occupations["conceptUri"].map(counts).fillna(0),
    )
    parts = []
    for job, keywords in TARGET_JOBS.items():
        mask = occupations["norm"].apply(
            lambda s: any(k in s for k in keywords) and not any(x in s for x in EXCLUDE)
        )
        found = occupations[mask].nlargest(MAX_PER_JOB, "n_skills")
        parts.append(found.assign(target_job=job))
    columns = ["target_job", "conceptUri", "preferredLabel", "n_skills"]
    return pd.concat(parts, ignore_index=True)[columns]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    occupations = load_esco("occupations", usecols=["conceptUri", "preferredLabel"])
    relations = load_esco("occupationSkillRelations")
    skills = load_esco(
        "skills", usecols=["conceptUri", "preferredLabel", "altLabels", "skillType"]
    )

    selected = select_occupations(occupations, relations)
    selected.to_csv(PROCESSED_DIR / "target_occupations.csv", index=False)

    rel = relations.merge(
        selected[["conceptUri", "target_job"]],
        left_on="occupationUri",
        right_on="conceptUri",
    )
    referential = (
        rel.groupby("skillUri")
        .agg(
            target_jobs=("target_job", lambda s: ", ".join(sorted(set(s)))),
            essential=("relationType", lambda s: (s == "essential").any()),
        )
        .reset_index()
        .merge(skills, left_on="skillUri", right_on="conceptUri", how="left")
        .drop(columns="conceptUri")
    )
    referential.to_csv(PROCESSED_DIR / "skills_referential.csv", index=False)

    print(selected.to_string(index=False))
    print(f"\n{len(referential)} compétences dans le référentiel")


if __name__ == "__main__":
    main()
