"""Réglage des pondérations du score par régression logistique (scikit-learn).

Les critères d'une paire (compétences, sens, expérience…) valent tous entre 0 et 1.
Une régression logistique apprend, sur des paires étiquetées, combien chaque
critère aide à prédire « pertinent » (label 2). Ses coefficients deviennent les
pondérations du moteur : le modèle reste lisible (pas de boîte noire) et le
score reste explicable critère par critère.

Règles :
- apprentissage sur les paires d'ENTRAÎNEMENT uniquement (jamais le test) ;
- un coefficient négatif (le critère n'aide pas) est ramené à 0 ;
- les pondérations sont normalisées (somme = 1), puis écrites dans un fichier
  de configuration séparé : les pondérations a priori restent intactes.
Attention : les pondérations apprises reflètent la façon dont les labels ont été
construits (labels provisoires : métier + expérience). Le choix final entre
pondérations a priori et apprises est une décision métier.
"""

from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

RELEVANT = 2


def learn_weights(parts: list[dict[str, float]], labels: list[int]) -> dict:
    """Pondérations apprises à partir des critères de paires étiquetées.

    Retourne {"weights": {...}, "coefficients": {...}} (coefficients bruts, pour
    le rapport)."""
    names = sorted(set.intersection(*(set(p) for p in parts)))  # critères communs
    X = np.array([[p[name] for name in names] for p in parts])
    y = np.array([label == RELEVANT for label in labels], dtype=int)
    if len(set(y)) < 2:
        raise ValueError("Il faut des paires pertinentes et non pertinentes")
    model = LogisticRegression(max_iter=1000).fit(X, y)
    coefficients = dict(zip(names, model.coef_[0]))
    positive = {name: max(0.0, float(c)) for name, c in coefficients.items()}
    total = sum(positive.values())
    if total == 0:
        raise ValueError("Aucun critère n'aide à prédire la pertinence")
    weights = {name: round(value / total, 4) for name, value in positive.items()}
    return {"weights": weights, "coefficients": coefficients}


def save_weights(weights: dict[str, float], path: Path, note: str) -> None:
    """Écrit les pondérations au format de config/matching.toml."""
    lines = [
        "# Pondérations APPRISES (scripts/tune_weights.py) — ne pas modifier à la main.",
        f"# {note}",
        f"# Générées le {date.today().isoformat()}.",
        "",
        "[weights]",
    ]
    lines += [f"{name} = {value}" for name, value in weights.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
