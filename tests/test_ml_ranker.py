import pytest

from ematching.matching.ml_ranker import learn_weights, save_weights
from ematching.matching.semantic import load_weights

# Paires fictives : la pertinence (label 2) dépend seulement de « skills ».
PARTS = [{"skills": s / 10, "location": (s * 7) % 2} for s in range(11)] * 5
LABELS = [2 if p["skills"] >= 0.5 else 0 for p in PARTS]


def test_learned_weights_favour_the_useful_criterion():
    weights = learn_weights(PARTS, LABELS)["weights"]
    assert weights["skills"] > 0.9
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_learning_needs_both_classes():
    with pytest.raises(ValueError):
        learn_weights(PARTS, [0] * len(PARTS))


def test_saved_weights_can_be_reloaded(tmp_path):
    path = tmp_path / "learned.toml"
    save_weights({"skills": 0.7, "experience": 0.3}, path, "test")
    assert load_weights(path) == pytest.approx({"skills": 0.7, "experience": 0.3})
