"""Évalue les moteurs de matching sur la grille de test (profils test x offres test).

Pour chaque profil de test, chaque moteur classe toutes les offres de test ; on
compare au label provisoire (test_grid.csv) : Precision@5, Recall@5, MRR, NDCG@5.

Moteurs comparés :
- recouvrement : part des compétences requises identiques (référence minimale) ;
- règles       : critères lisibles (compétences identiques, expérience, diplôme,
                 ville), pondérations de config/matching.toml, sans embeddings ;
- hybride      : règles + compétences équivalentes par le sens + proximité
                 globale (modèle d'embeddings, option --models).

Précaution : les colonnes qui servent à calculer les labels (métier source,
métier visé) sont retirées avant de donner profils et offres aux moteurs.

Usage :
  python scripts/evaluate_matching.py                       # modèle par défaut
  python scripts/evaluate_matching.py --models e5-small qwen3-0.6b
  python scripts/evaluate_matching.py --models none         # sans embeddings
  python scripts/evaluate_matching.py --example P0003       # Top 5 expliqué
  python scripts/evaluate_matching.py --learned             # + pondérations apprises
    (fichiers config/matching_learned_<modèle>.toml créés par scripts/tune_weights.py)
"""

import argparse
import time

import numpy as np
import pandas as pd

from ematching.data import PROCESSED_DIR
from ematching.embeddings import DEFAULT_MODEL
from ematching.evaluation import RELEVANT, evaluate_rankings
from ematching.features import FORBIDDEN
from ematching.matching.semantic import SemanticEngine, load_weights
from tune_weights import learned_path


def load_test_data():
    grid = pd.read_csv(PROCESSED_DIR / "test_grid.csv")
    profiles = pd.read_csv(PROCESSED_DIR / "profiles.csv").drop(
        columns=list(FORBIDDEN), errors="ignore"
    )
    offers = pd.read_csv(PROCESSED_DIR / "offers.csv").drop(
        columns=list(FORBIDDEN), errors="ignore"
    )
    profiles = profiles[profiles["profile_id"].isin(grid["profile_id"])]
    offers = offers[offers["offer_id"].isin(grid["offer_id"])]
    labels = {
        (p, o): label
        for p, o, label in zip(
            grid["profile_id"], grid["offer_id"], grid["silver_label"]
        )
    }
    return profiles.to_dict("records"), offers.to_dict("records"), labels


def run(engine, profiles, offers, labels) -> tuple[dict, float]:
    """Mesures moyennes et temps moyen (secondes) pour classer un profil."""
    rankings, start = [], time.perf_counter()
    for profile in profiles:
        results = engine.rank(profile, offers, k=len(offers))  # classement complet
        rankings.append([labels[(profile["profile_id"], r.offer_id)] for r in results])
    seconds = (time.perf_counter() - start) / len(profiles)
    return evaluate_rankings(rankings), seconds


def show_example(engine, profile_id, profiles, offers, labels) -> None:
    profile = next(p for p in profiles if p["profile_id"] == profile_id)
    print(f"\nTop 5 pour {profile_id} ({engine.name}) — compétences du profil :")
    print(f"  {profile['skills'].replace('|', ', ')}")
    print(
        f"  {profile['years_experience']} an(s), {profile['education_level']}, "
        f"{profile['city']}"
    )
    for rank, result in enumerate(engine.rank(profile, offers), start=1):
        offer = next(o for o in offers if o["offer_id"] == result.offer_id)
        label = labels[(profile_id, result.offer_id)]
        print(
            f"\n  {rank}. {result.offer_id} « {offer['title']} » — score "
            f"{result.score:.2f} (label {label})"
        )
        print(f"     critères   : {result.parts}")
        print(f"     communes   : {result.common_skills}")
        print(f"     manquantes : {result.missing_skills}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du matching")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--example", help="identifiant d'un profil de test")
    parser.add_argument(
        "--learned",
        action="store_true",
        help="compare aussi les pondérations apprises (scripts/tune_weights.py)",
    )
    parser.add_argument(
        "--skill-threshold",
        type=float,
        help="seuil d'équivalence pour un modèle hors embeddings.MODEL_SPECS",
    )
    args = parser.parse_args()
    profiles, offers, labels = load_test_data()

    engines = [("recouvrement", SemanticEngine(weights={"skills": 1.0}))]
    models = ["none"] + [m for m in args.models if m.lower() != "none"]
    for name in models:
        encoder = None
        if name.lower() != "none":
            from ematching.embeddings import SentenceEncoder

            encoder = SentenceEncoder(name)  # chargé une seule fois par modèle
        label = "règles" if encoder is None else f"hybride {name}"
        configs = [("a priori", load_weights())]
        if args.learned and learned_path(name).exists():
            configs.append(("appris", load_weights(learned_path(name))))
        elif args.learned:
            print(f"(pas de pondérations apprises pour {name} : lancer tune_weights)")
        for config, weights in configs:
            engine = SemanticEngine(encoder, args.skill_threshold, weights)
            engines.append((f"{label} ({config})", engine))

    # Meilleure Precision@5 possible : peu de profils ont 5 offres pertinentes.
    n_relevant = [
        sum(labels[(p["profile_id"], o["offer_id"])] == RELEVANT for o in offers)
        for p in profiles
    ]
    ceiling = np.mean([min(n, 5) / 5 for n in n_relevant])
    print(
        f"{len(profiles)} profils x {len(offers)} offres de test ; "
        f"P@5 maximale possible = {ceiling:.3f}\n"
    )
    print(
        f"{'moteur':28} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'NDCG@5':>7} {'s/profil':>9}"
    )
    for name, engine in engines:
        scores, seconds = run(engine, profiles, offers, labels)
        print(
            f"{name:28} {scores['P@5']:6.3f} {scores['R@5']:6.3f} "
            f"{scores['MRR']:6.3f} {scores['NDCG@5']:7.3f} {seconds:9.3f}"
        )

    if args.example:
        show_example(engines[-1][1], args.example, profiles, offers, labels)


if __name__ == "__main__":
    main()
