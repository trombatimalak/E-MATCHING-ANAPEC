from ematching.annotation import agreement


def test_agreement_is_perfect_for_identical_labels():
    labels = {"p1": 0, "p2": 1, "p3": 2, "p4": 2}
    result = agreement(labels, labels)
    assert result["n"] == 4
    assert result["accuracy"] == 1.0
    assert result["kappa"] == 1.0


def test_agreement_counts_only_common_pairs():
    result = agreement({"p1": 0, "p2": 1}, {"p2": 1, "p3": 2})
    assert result["n"] == 1
