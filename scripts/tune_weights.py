"""Apprend les pondérations du score sur les paires d'ENTRAÎNEMENT (pairs.csv).

Les critères de chaque paire sont calculés par le moteur (le même que pour le
classement), puis une régression logistique en déduit les pondérations
(voir matching/ml_ranker.py). Résultat : config/matching_learned_<modèle>.toml,
à comparer aux pondérations a priori avec :
  python scripts/evaluate_matching.py --models e5-small --learned

Usage :
  python scripts/tune_weights.py --model e5-small
  python scripts/tune_weights.py --model none        # sans embeddings
"""

import argparse

import pandas as pd

from ematching.data import PROCESSED_DIR, PROJECT_ROOT
from ematching.embeddings import DEFAULT_MODEL
from ematching.features import FORBIDDEN
from ematching.matching.ml_ranker import learn_weights, save_weights
from ematching.matching.semantic import SemanticEngine


def learned_path(model: str):
    safe = model.replace("/", "_").replace("\\", "_")
    return PROJECT_ROOT / "config" / f"matching_learned_{safe}.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Réglage des pondérations")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modèle ou « none »")
    parser.add_argument(
        "--skill-threshold",
        type=float,
        help="seuil d'équivalence pour un modèle hors embeddings.MODEL_SPECS",
    )
    args = parser.parse_args()

    pairs = pd.read_csv(PROCESSED_DIR / "pairs.csv")
    pairs = pairs[pairs["split"] == "train"]  # jamais les paires de test
    drop = {"columns": list(FORBIDDEN), "errors": "ignore"}
    profiles = pd.read_csv(PROCESSED_DIR / "profiles.csv").drop(**drop)
    offers = pd.read_csv(PROCESSED_DIR / "offers.csv").drop(**drop)
    profiles = profiles.set_index("profile_id", drop=False).to_dict("index")
    offers = offers.set_index("offer_id", drop=False).to_dict("index")

    encoder = None
    if args.model.lower() != "none":
        from ematching.embeddings import SentenceEncoder

        encoder = SentenceEncoder(args.model)
    engine = SemanticEngine(encoder, args.skill_threshold)

    parts = [
        engine.score(profiles[p], offers[o]).parts
        for p, o in zip(pairs["profile_id"], pairs["offer_id"])
    ]
    result = learn_weights(parts, pairs["silver_label"].tolist())

    print(f"{len(parts)} paires d'entraînement, modèle : {args.model}\n")
    print(f"{'critère':14} {'coefficient':>12} {'pondération':>12}")
    for name, coefficient in result["coefficients"].items():
        print(f"{name:14} {coefficient:12.3f} {result['weights'][name]:12.3f}")
    path = learned_path(args.model)
    save_weights(
        result["weights"],
        path,
        f"Régression logistique sur {len(parts)} paires d'entraînement, "
        f"labels provisoires, modèle {args.model}.",
    )
    print(f"\nPondérations écrites dans {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
