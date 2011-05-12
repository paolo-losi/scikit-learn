import numpy as np
from numpy.testing import assert_almost_equal

from ..platt import fit_platt_logreg, PlattScaler
from ...svm import SVC
from ...cross_val import KFold


def test_fit_platt_simple():
    """test platt optimization on a simple hand calculated case"""

    a, b = fit_platt_logreg([-1, 1], [0, 1])
    assert_almost_equal(a, np.log(2), decimal=5)
    assert_almost_equal(b, 0, decimal=5)


def test_fit_plat_synthetic():
    """test platt calibration on a synthetic case"""

    test_a = 2.
    test_b = -4.

    # since we have as high as 1e4 samples (n_score * sample_per_score)
    # distributed evenly between two classes,
    # bayesian correction in platt scaling is negligible and
    # platt scaling falls back to standard logistic regression fitting
    sample_per_score = 1000
    n_score = 10

    score_  = np.linspace(1., 3, n_score)
    outcome = np.zeros((n_score * sample_per_score))
    score   = np.empty((n_score * sample_per_score))
    proba_ = 1. / (1. + np.exp(-(test_a * score_ + test_b)))
    for i in range(0, n_score * sample_per_score, n_score):
        proba_limit = (i + n_score / 2.) / (n_score * sample_per_score)
        outcome[i: i + n_score][proba_ > proba_limit] = 1
        score  [i: i + n_score] = score_

    a, b = fit_platt_logreg(score, outcome)
    assert_almost_equal(a, test_a, decimal=2)
    assert_almost_equal(b, test_b, decimal=2)


def test_platt_scaler():
    """compare PlattScaler with libsvm builtin scaler"""
    X = np.array([0., 0.5, 1., 1.2, 2.] * 20).reshape((-1, 1))
    y = np.array([0,  0,   1,  0,   1 ] * 20)

    svc_proba = SVC(probability=True, C=100.)
    ref_proba = svc_proba.fit(X, y).predict_proba(1.3)

    cv = KFold(y.size, k=5)  # same cross val as libsvm
    platt_scaler = PlattScaler(SVC(C=100.))
    proba = platt_scaler.fit(X, y, cv=cv).predict_proba(1.3)

    assert_almost_equal(proba, ref_proba, decimal=3)
