"""Annote les paires dans le terminal (sauvegarde après chaque réponse)."""

import sys

import pandas as pd

LIMIT = 100
LABEL_COLUMNS = {"label_annotator_a": "object", "label_annotator_b": "object"}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage : python scripts/annotate_cli.py fichier.csv [colonne]")
    path = sys.argv[1]
    column = sys.argv[2] if len(sys.argv) > 2 else "label_annotator_a"
    data = pd.read_csv(path, dtype=LABEL_COLUMNS)

    for index in range(min(LIMIT, len(data))):
        row = data.iloc[index]
        if pd.notna(row[column]) and str(row[column]).strip():
            continue
        print(f"\n--- Paire {index + 1}/{LIMIT} ({row['pair_id']}) ---")
        print(f"PROFIL : {row['cv_text']}")
        print(
            f"OFFRE  : {row['title']} | {row['city']} | {row['education_level']} "
            f"| expérience min : {row['min_experience_years']} an(s)"
        )
        print(f"         {row['description']}")
        answer = ""
        while answer not in {"0", "1", "2", "q"}:
            answer = input("Label (0/1/2, q pour quitter) : ").strip().lower()
        if answer == "q":
            break
        data.at[index, column] = answer
        data.to_csv(path, index=False)

    done = data[column].notna().sum()
    print(f"\n{done} paires annotées dans {path}")


if __name__ == "__main__":
    main()
