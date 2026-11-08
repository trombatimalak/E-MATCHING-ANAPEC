"""Construit les paires profil-offre avec un label de pertinence provisoire."""

import random

import pandas as pd

from ematching.data import PROCESSED_DIR

SEED = 42
TEST_SHARE = 0.3
N_ANNOTATION = 400
SEP = "|"


def assign_split(df: pd.DataFrame, id_col: str, job_col: str, rng: random.Random):
    """Sépare les identifiants en train/test, de façon stratifiée par métier."""
    split = {}
    for _, group in df.groupby(job_col):
        ids = group[id_col].tolist()
        rng.shuffle(ids)
        n_test = round(len(ids) * TEST_SHARE)
        for position, identifier in enumerate(ids):
            split[identifier] = "test" if position < n_test else "train"
    return split


def overlap(profile_skills: set, required_skills: set) -> float:
    """Part des compétences requises couvertes par le profil."""
    if not required_skills:
        return 0.0
    return len(profile_skills & required_skills) / len(required_skills)


def silver_label(profile: dict, offer: dict) -> int:
    """0 = non pertinent, 1 = partiellement pertinent, 2 = pertinent."""
    same_occupation = profile["source_occupation_uri"] == offer["source_occupation_uri"]
    same_job = profile["desired_job"] == offer["target_job"]
    experience_ok = profile["years_experience"] >= offer["min_experience_years"]
    if same_occupation and experience_ok:
        return 2
    if same_occupation or same_job:
        return 1
    return 0


def sample(items: list, k: int, rng: random.Random) -> list:
    return rng.sample(items, min(k, len(items)))


def pairs_for_profile(profile: dict, offers: list, rng: random.Random) -> list:
    own_skills = set(profile["skills"].split(SEP))
    scored = [
        (o, overlap(own_skills, set(o["required_skills"].split(SEP)))) for o in offers
    ]
    same_occ = [
        x
        for x in scored
        if x[0]["source_occupation_uri"] == profile["source_occupation_uri"]
    ]
    same_job = [
        x
        for x in scored
        if x[0]["target_job"] == profile["desired_job"] and x not in same_occ
    ]
    others = [x for x in scored if x[0]["target_job"] != profile["desired_job"]]
    hard = sorted(others, key=lambda x: -x[1])[:5]

    picks = (
        [("same_occupation", x) for x in sample(same_occ, 2, rng)]
        + [("same_job", x) for x in sample(same_job, 1, rng)]
        + [("hard_negative", x) for x in sample(hard, 2, rng)]
        + [("easy_negative", x) for x in sample(others, 1, rng)]
    )
    rows, seen = [], set()
    for pair_type, (offer, score) in picks:
        if offer["offer_id"] in seen:
            continue
        seen.add(offer["offer_id"])
        rows.append(
            {
                "profile_id": profile["profile_id"],
                "offer_id": offer["offer_id"],
                "pair_type": pair_type,
                "skill_overlap": round(score, 3),
                "silver_label": silver_label(profile, offer),
            }
        )
    return rows


def main() -> None:
    rng = random.Random(SEED)
    profiles = pd.read_csv(PROCESSED_DIR / "profiles.csv")
    offers = pd.read_csv(PROCESSED_DIR / "offers.csv")

    profile_split = assign_split(profiles, "profile_id", "desired_job", rng)
    offer_split = assign_split(offers, "offer_id", "target_job", rng)

    rows = []
    for split in ("train", "test"):
        split_offers = [
            o for o in offers.to_dict("records") if offer_split[o["offer_id"]] == split
        ]
        for prof in profiles.to_dict("records"):
            if profile_split[prof["profile_id"]] == split:
                for row in pairs_for_profile(prof, split_offers, rng):
                    rows.append({**row, "split": split})

    pairs = pd.DataFrame(rows)
    pairs.insert(0, "pair_id", [f"PR{i + 1:05d}" for i in range(len(pairs))])
    pairs.to_csv(PROCESSED_DIR / "pairs.csv", index=False)

    # Grille complète du split test, pour évaluer le ranking (Top-5).
    test_profiles = profiles[profiles["profile_id"].map(profile_split) == "test"]
    test_offers = offers[offers["offer_id"].map(offer_split) == "test"]
    grid = [
        {
            "profile_id": prof["profile_id"],
            "offer_id": offer["offer_id"],
            "silver_label": silver_label(prof, offer),
        }
        for prof in test_profiles.to_dict("records")
        for offer in test_offers.to_dict("records")
    ]
    pd.DataFrame(grid).to_csv(PROCESSED_DIR / "test_grid.csv", index=False)
    print(f"Grille de test : {len(grid)} paires")

    # Échantillon à annoter : sans le label provisoire ni le type de paire.
    sample_ids = pairs.sample(min(N_ANNOTATION, len(pairs)), random_state=SEED)
    sheet = (
        sample_ids[["pair_id", "profile_id", "offer_id"]]
        .merge(profiles[["profile_id", "cv_text"]], on="profile_id")
        .merge(
            offers[
                [
                    "offer_id",
                    "title",
                    "description",
                    "city",
                    "education_level",
                    "min_experience_years",
                ]
            ],
            on="offer_id",
        )
    )
    sheet["label_annotator_a"] = ""
    sheet["label_annotator_b"] = ""
    sheet.to_csv(PROCESSED_DIR / "annotation_sample.csv", index=False)

    print(pairs.groupby(["split", "silver_label"]).size().unstack(fill_value=0))
    print()
    print(pairs.groupby(["split", "pair_type"]).size().unstack(fill_value=0))
    print(f"\n{len(pairs)} paires, {len(sheet)} à annoter")


if __name__ == "__main__":
    main()
