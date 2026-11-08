"""Compare les annotations humaines aux labels provisoires et entre annotateurs."""

import sys

import pandas as pd

from ematching.annotation import agreement, read_labels
from ematching.data import PROCESSED_DIR


def show(title: str, result: dict) -> None:
    print(f"\n{title}")
    for key, value in result.items():
        print(f"  {key} : {round(value, 3) if isinstance(value, float) else value}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(
            "Usage : python scripts/annotation_agreement.py fichier1.csv [fichier2.csv]"
        )
    pairs = pd.read_csv(PROCESSED_DIR / "pairs.csv").set_index("pair_id")
    silver = pairs["silver_label"]

    first = read_labels(sys.argv[1])
    show("Annotateur 1 vs labels provisoires", agreement(first, silver))
    print()
    print(
        pd.crosstab(
            first, silver.loc[first.index], rownames=["humain"], colnames=["provisoire"]
        )
    )

    if len(sys.argv) > 2:
        second = read_labels(sys.argv[2])
        show("Annotateur 1 vs annotateur 2", agreement(first, second))


if __name__ == "__main__":
    main()
