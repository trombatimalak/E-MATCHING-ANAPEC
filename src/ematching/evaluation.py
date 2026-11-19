"""Mesures de qualité d'un classement (Top-k), pour comparer les moteurs.

Pour un profil, on classe des offres ; chaque offre a un label de pertinence :
2 = pertinente, 1 = partiellement pertinente, 0 = non pertinente.
- Precision@k : part des k premières offres qui sont pertinentes (label 2).
- Recall@k    : part des offres pertinentes qui sont dans les k premières.
- MRR         : 1 / rang de la première offre pertinente (1 si elle est 1re).
- NDCG@k      : qualité de l'ordre en tenant compte des labels 0/1/2 ; vaut 1
                quand le classement est le meilleur possible.
Les fonctions prennent les labels dans l'ordre du classement proposé.
"""

import math

RELEVANT = 2
K = 5


def precision_at_k(ranked: list[int], k: int = K) -> float:
    return sum(label == RELEVANT for label in ranked[:k]) / k


def recall_at_k(ranked: list[int], k: int = K) -> float | None:
    """None si le profil n'a aucune offre pertinente (non comptabilisé)."""
    n_relevant = sum(label == RELEVANT for label in ranked)
    if n_relevant == 0:
        return None
    return sum(label == RELEVANT for label in ranked[:k]) / n_relevant


def reciprocal_rank(ranked: list[int]) -> float:
    for position, label in enumerate(ranked, start=1):
        if label == RELEVANT:
            return 1 / position
    return 0.0


def ndcg_at_k(ranked: list[int], k: int = K) -> float:
    def dcg(labels):
        return sum((2**label - 1) / math.log2(i + 2) for i, label in enumerate(labels))

    ideal = dcg(sorted(ranked, reverse=True)[:k])
    return dcg(ranked[:k]) / ideal if ideal else 0.0


def evaluate_rankings(rankings: list[list[int]], k: int = K) -> dict[str, float]:
    """Moyennes sur tous les profils ; chaque élément = labels d'un classement
    complet (toutes les offres candidates, dans l'ordre proposé)."""
    recalls = [
        r for r in (recall_at_k(ranked, k) for ranked in rankings) if r is not None
    ]
    n = len(rankings)
    return {
        f"P@{k}": sum(precision_at_k(ranked, k) for ranked in rankings) / n,
        f"R@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "MRR": sum(reciprocal_rank(ranked) for ranked in rankings) / n,
        f"NDCG@{k}": sum(ndcg_at_k(ranked, k) for ranked in rankings) / n,
    }
