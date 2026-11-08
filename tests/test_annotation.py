import pandas as pd

from ematching.annotation import agreement


def test_agreement_is_perfect_for_identical_labels():
    labels = pd.Series([0, 1, 2, 2], index=["p1", "p2", "p3", "p4"])
    result = agreement(labels, labels)
    assert result["accuracy"] == 1.0
    assert result["kappa"] == 1.0
