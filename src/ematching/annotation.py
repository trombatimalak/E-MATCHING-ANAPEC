"""Lecture des annotations humaines et mesures d'accord (bibliothèque standard)."""

import csv

K = 3  # labels possibles : 0, 1, 2
NAN = float("nan")


def read_labels(path) -> dict[str, int]:
    """Retourne {pair_id: label} pour les lignes annotées (colonne a ou b)."""
    labels = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = (
                row.get("label_annotator_a") or row.get("label_annotator_b") or ""
            ).strip()
            if not value:
                continue
            label = int(float(value))
            if label not in range(K):
                raise ValueError(f"Label invalide : {label} ({row['pair_id']})")
            labels[row["pair_id"]] = label
    return labels


def confusion_matrix(first: dict, second: dict) -> list[list[int]]:
    """Matrice : lignes = labels de `first`, colonnes = labels de `second`."""
    matrix = [[0] * K for _ in range(K)]
    for pair_id in first.keys() & second.keys():
        matrix[first[pair_id]][second[pair_id]] += 1
    return matrix


def agreement(first: dict, second: dict) -> dict:
    """Accord brut, kappa de Cohen et kappa pondéré (quadratique)."""
    matrix = confusion_matrix(first, second)
    n = sum(sum(row) for row in matrix)
    if n == 0:
        return {"n": 0, "accuracy": NAN, "kappa": NAN, "kappa_weighted": NAN}

    rows = [sum(matrix[a]) for a in range(K)]
    cols = [sum(matrix[a][b] for a in range(K)) for b in range(K)]
    accuracy = sum(matrix[k][k] for k in range(K)) / n
    chance = sum(rows[k] * cols[k] for k in range(K)) / n**2
    kappa = (accuracy - chance) / (1 - chance) if chance < 1 else NAN

    weights = [[(a - b) ** 2 / (K - 1) ** 2 for b in range(K)] for a in range(K)]
    observed = sum(weights[a][b] * matrix[a][b] for a in range(K) for b in range(K))
    expected = sum(
        weights[a][b] * rows[a] * cols[b] / n for a in range(K) for b in range(K)
    )
    kappa_weighted = 1 - observed / expected if expected else NAN
    return {
        "n": n,
        "accuracy": accuracy,
        "kappa": kappa,
        "kappa_weighted": kappa_weighted,
    }
