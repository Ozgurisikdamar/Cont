import numpy as np
import pytest
from scipy.special import softmax

from contextlens.evaluation.metrics import (
    expected_calibration_error,
    multiclass_report,
    multilabel_report,
    ood_report,
)
from contextlens.models.heads import calibrated_softmax, decide_subtopics, fit_temperature, joint_subtopic_probs


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 2000, 4
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.0
    logits *= 6.0  # grossly over-confident
    t = fit_temperature(logits, y)
    assert t > 2.0
    before = expected_calibration_error(softmax(logits, axis=1), y)
    after = expected_calibration_error(calibrated_softmax(logits, t), y)
    assert after < before / 2


def test_ece_zero_for_perfect_confident_predictions():
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert expected_calibration_error(probs, np.array([0, 1, 2, 0])) == pytest.approx(0.0)


def test_multiclass_report_fields():
    rep = multiclass_report(np.array([0, 1, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]), ["a", "b"])
    assert rep["accuracy"] == pytest.approx(2 / 3)
    assert rep["confusion_matrix"] == [[1, 0], [1, 1]]
    assert set(rep["per_class"]) == {"a", "b"}


def test_decide_subtopics_respects_hierarchy():
    parent = np.array([0, 0, 1, 1])
    probs = np.array([[0.2, 0.6, 0.9, 0.9], [0.1, 0.2, 0.3, 0.8]])
    dec = decide_subtopics(probs, parent, np.array([0, 1]), threshold=0.5)
    assert dec.tolist() == [[0, 1, 0, 0], [0, 0, 0, 1]]  # never a child of another general topic
    dec = decide_subtopics(probs, parent, np.array([1, 1]), threshold=0.25)
    assert dec[1].tolist() == [0, 0, 1, 1]  # multi-label above threshold


def test_joint_probs():
    gp = np.array([[0.7, 0.3]])
    cond = np.array([[0.5, 0.5, 1.0, 0.0]])
    assert joint_subtopic_probs(gp, cond, np.array([0, 0, 1, 1])).tolist() == [[0.35, 0.35, 0.3, 0.0]]


def test_multilabel_report():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    rep = multilabel_report(y, y.copy(), y.astype(float), ["a", "b", "c"])
    assert rep["subset_accuracy"] == 1.0 and rep["hamming_loss"] == 0.0 and rep["precision_at_1"] == 1.0


def test_ood_report_separable():
    rep = ood_report(np.array([0.9, 0.8, 0.95]), np.array([0.1, 0.2]))
    assert rep["auroc"] == 1.0 and rep["fpr_at_95_tpr"] == 0.0
