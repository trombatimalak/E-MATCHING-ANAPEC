"""Accord entre annotations humaines, et avec les labels provisoires."""

import csv
import sys
from pathlib import Path

from ematching.annotation import agreement, confusion_matrix, read_labels

PAIRS = Path(__file__).resolve().parents[1] / "data" / "processed" / "pairs.csv"


def read_silver() -> dict[str, int]:
    with open(PAIRS, encoding="utf-8", newline="") as handle:
        return {r["pair_id"]: int(r["silver_label"]) for r in csv.DictReader(handle)}


def show(title: str, first: dict, second: dict, names: tuple[str, str]) -> None:
    result = agreement(first, second)
    print(f"\n=== {title} ===")
    print(f"Paires comparées : {result['n']}")
    print(f"Accord brut      : {result['accuracy']:.3f}")
    print(f"Kappa            : {result['kappa']:.3f}")
    print(f"Kappa pondéré    : {result['kappa_weighted']:.3f}")
    print(f"\nLignes = {names[0]}, colonnes = {names[1]}")
    for label, row in enumerate(confusion_matrix(first, second)):
        print(f"  {label} : " + "  ".join(f"{count:4d}" for count in row))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage : python scripts/annotation_agreement.py f1.csv [f2.csv]")
    first = read_labels(sys.argv[1])
    show(
        "Annotateur 1 vs labels provisoires",
        first,
        read_silver(),
        ("humain", "provisoire"),
    )
    if len(sys.argv) > 2:
        second = read_labels(sys.argv[2])
        show(
            "Annotateur 1 vs annotateur 2",
            first,
            second,
            ("annotateur 1", "annotateur 2"),
        )


if __name__ == "__main__":
    main()
