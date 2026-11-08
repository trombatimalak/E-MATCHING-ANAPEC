"""Contrôle qualité du dataset (à lancer après chaque régénération)."""

import sys

import pandas as pd

from ematching.data import PROCESSED_DIR
from ematching.preprocessing import find_issues


def main() -> None:
    profiles = pd.read_csv(PROCESSED_DIR / "profiles.csv")
    offers = pd.read_csv(PROCESSED_DIR / "offers.csv")
    pairs = pd.read_csv(PROCESSED_DIR / "pairs.csv")
    test_grid = pd.read_csv(PROCESSED_DIR / "test_grid.csv")

    issues = find_issues(profiles, offers, pairs, test_grid)
    print(f"{len(profiles)} profils, {len(offers)} offres, {len(pairs)} paires")
    if issues:
        print("PROBLÈMES :")
        for issue in issues:
            print(f" - {issue}")
        sys.exit(1)
    print("Dataset OK : aucun problème détecté")


if __name__ == "__main__":
    main()
