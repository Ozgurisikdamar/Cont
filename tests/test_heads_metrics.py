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


def test_macro_f1_over_supported_labels_ignores_absent_classes():
    import numpy as np

    from contextlens.evaluation.metrics import multilabel_report

    y = np.array([[1, 0, 0], [0, 1, 0]])  # label 2 never occurs in this evaluation set
    rep = multilabel_report(y, y.copy(), y.astype(float), ["a", "b", "c"])
    assert rep["macro_f1"] < 1.0  # the absent label counts as F1 = 0
    assert rep["macro_f1_supported"] == 1.0 and rep["labels_supported"] == 2


def test_grouped_probs_are_distributions_and_respect_the_hierarchy():
    from contextlens.models.heads import grouped_probs

    parent = np.array([0, 0, 1, 1, 1])
    logits = np.array([[2.0, 1.0, 0.0, -1.0, -np.inf], [0.0, 0.0, 3.0, 3.0, 1.0]])
    general, cond = grouped_probs(logits, 1.0, parent, 2)
    assert np.allclose(general.sum(axis=1), 1.0)
    for g in (0, 1):  # conditionals of each general topic sum to 1
        assert np.allclose(cond[:, parent == g].sum(axis=1), 1.0)
    assert cond[0, 4] == 0.0  # a subtopic never seen in training (-inf logit) gets probability 0
    assert general[1].argmax() == 1


def test_grouped_temperature_recovers_the_generating_temperature():
    from contextlens.models.heads import fit_grouped_temperature, grouped_probs

    rng = np.random.default_rng(42)
    parent = np.repeat(np.arange(4), 3)
    logits = rng.normal(0, 3, size=(20000, 12))
    general, _ = grouped_probs(logits, 2.0, parent, 4)
    y = np.array([rng.choice(4, p=p) for p in general])
    assert abs(fit_grouped_temperature(logits, y, parent) - 2.0) < 0.15
