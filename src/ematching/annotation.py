"""Lecture des annotations humaines et calcul des accords."""

import pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID_LABELS = {0, 1, 2}


def read_labels(path) -> pd.Series:
    """Retourne {pair_id: label} pour les lignes annotées d'un fichier."""
    data = pd.read_csv(path)
    labels = data["label_annotator_a"].fillna(data["label_annotator_b"])
    data = data.assign(label=pd.to_numeric(labels, errors="coerce"))
    data = data.dropna(subset=["label"])
    series = data.set_index("pair_id")["label"].astype(int)
    invalid = set(series) - VALID_LABELS
    if invalid:
        raise ValueError(f"Labels invalides (attendu 0, 1 ou 2) : {sorted(invalid)}")
    return series


def agreement(first: pd.Series, second: pd.Series) -> dict:
    """Accord entre deux séries de labels sur leurs identifiants communs."""
    common = first.index.intersection(second.index)
    if len(common) == 0:
        nan = float("nan")
        return {"n": 0, "accuracy": nan, "kappa": nan, "kappa_quadratic": nan}
    a, b = first.loc[common], second.loc[common]
    return {
        "n": len(common),
        "accuracy": float((a == b).mean()),
        "kappa": float(cohen_kappa_score(a, b)),
        "kappa_quadratic": float(cohen_kappa_score(a, b, weights="quadratic")),
    }
