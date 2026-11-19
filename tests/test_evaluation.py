import math

import pytest

from ematching.evaluation import (
    evaluate_rankings,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Labels des offres dans l'ordre proposé par un moteur (2 = pertinente).
RANKED = [0, 2, 1, 0, 2, 0, 2]


def test_precision_and_recall_at_5():
    assert precision_at_k(RANKED) == 2 / 5  # 2 pertinentes dans le Top 5
    assert recall_at_k(RANKED) == 2 / 3  # 2 des 3 pertinentes
    assert recall_at_k([0, 1, 0]) is None  # aucune offre pertinente : ignoré


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED) == 1 / 2  # 1re pertinente au rang 2
    assert reciprocal_rank([0, 1]) == 0.0


def test_ndcg_hand_computed():
    # DCG = (2²-1)/log2(3) + (2¹-1)/log2(4) ; idéal = 3/log2(2) + 1/log2(3)
    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([0, 2, 1]) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([2, 1, 0]) == pytest.approx(1.0)  # ordre parfait


def test_evaluate_rankings_averages_profiles():
    scores = evaluate_rankings([[2, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert scores["P@5"] == pytest.approx(0.1)
    assert scores["R@5"] == pytest.approx(1.0)  # 2e profil sans pertinente ignoré
    assert scores["MRR"] == pytest.approx(0.5)
