"""Démonstration : un CV -> les 5 meilleures offres, expliquées.

Usage :
  python scripts/demo_cv.py data/samples/cv/cv_03_comptable.pdf
  python scripts/demo_cv.py mon_cv.docx --model qwen3-0.6b
  python scripts/demo_cv.py mon_cv.pdf --model none          # sans embeddings
  python scripts/demo_cv.py mon_cv.pdf --weights config/matching_learned_e5-small.toml
"""

import argparse

from ematching.cv_parser import UnsupportedCVError
from ematching.embeddings import DEFAULT_MODEL
from ematching.prediction import Recommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Top 5 des offres pour un CV")
    parser.add_argument("cv", help="fichier PDF ou DOCX")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modèle ou « none »")
    parser.add_argument("--weights", help="fichier de pondérations (.toml)")
    parser.add_argument("-k", type=int, default=5, help="nombre d'offres")
    args = parser.parse_args()

    recommender = Recommender(model=args.model, weights_path=args.weights)
    try:
        result = recommender.recommend(args.cv, k=args.k)
    except UnsupportedCVError as error:
        print(f"CV non pris en charge : {error}")
        return

    profile = result.profile
    print(f"\nPROFIL EXTRAIT — {args.cv}")
    print(
        f"  Études : {profile['education_level']} | Expérience : "
        f"{profile['years_experience']} an(s) | Ville : {profile['city']} | "
        f"Langues : {profile['languages']}"
    )
    print(f"\nCOMPÉTENCES RECONNUES ({len(result.skills)})")
    for match in result.skills:
        print(f"  - {match.label}  [{match.method}]  <- « {match.source[:60]} »")
    for warning in result.warnings:
        print(f"  ! {warning}")

    print(f"\nTOP {args.k} DES OFFRES")
    for rank, (match, offer) in enumerate(zip(result.results, result.offers), start=1):
        print(
            f"\n{rank}. {offer['title']} — {offer['city']}, {offer['contract_type']} "
            f"({offer['offer_id']}) — score {match.score:.2f}"
        )
        print(f"   Détail       : {match.parts}")
        print(f"   En commun    : {', '.join(match.common_skills) or '—'}")
        print(f"   À développer : {', '.join(match.missing_skills) or '—'}")
    print("\nLe score est une aide à la recommandation, pas une décision.")


if __name__ == "__main__":
    main()
