import numpy as np

from scipy.optimize import fmin_bfgs


def fit_platt_logreg(score, y):
    y = np.asanyarray(y).ravel()
    score = np.asanyarray(score, dtype=np.float64).ravel()

    uniq = np.sort(np.unique(y))
    if np.size(uniq) != 2:
        raise ValueError('only binary classification is supported. classes: %s'
                                                                        % uniq)

    # the score is standardized to make logloss and ddx_logloss
    # numerically stable
    score_std = score.std()
    score_mean = score.mean()
    score_std = 1
    score_mean = 0
    n_score = (score - score_mean) / score_std

    n = y == uniq[0]
    p = y == uniq[1]
    n_n = float(np.sum(n))
    n_p = float(np.sum(p))
    yy = np.empty(y.shape, dtype=np.float64)
    yy[n] = 1. / (2. + n_n)
    yy[p] = (1. + n_p) / (2. + n_p)
    one_minus_yy = 1 - yy

    def logloss(x):
        a, b = x
        z = a * n_score + b
        ll_p = np.log1p(np.exp(-z))
        ll_n = np.log1p(np.exp(z))
        #print "logloss" + "-"*70
        #print "a", a, "b", b, "z", z, "ll_p", ll_p, "ll_n", ll_n
        return (one_minus_yy * ll_n + yy * ll_p).sum()

    def ddx_logloss(x):
        a, b = x
        z = a * n_score + b
        exp_z   = np.exp(z)
        exp_m_z = np.exp(-z)
        dda_ll_p = -n_score / (1 + exp_z)
        dda_ll_n =  n_score / (1 + exp_m_z)
        ddb_ll_p = -1 / (1 + exp_z)
        ddb_ll_n =  1 / (1 + exp_m_z)
        gradient = np.array([(one_minus_yy * dda_ll_n + yy * dda_ll_p).sum(),
                             (one_minus_yy * ddb_ll_n + yy * ddb_ll_p).sum()])
        #print "ddx_logloss" + "-"*70
        #print "a", a, "b", b, "gradient", gradient
        return gradient

    # FIXME check if fmin_bfgs converges
    a, b = fmin_bfgs(logloss, [0, 0], ddx_logloss)
    return a / score_std, b - a * score_mean / score_std


class PlattScaler(Classifier?):
    """Predicting Good Probabilities With Supervised Learning"""

    def __init__(self, clf):
        self.clf

    def fit(self, X, y, cv=None, **params):
        np.concatenate([...])
        pass

    def predict_proba(self, y):
        pass


