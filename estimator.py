import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_ate_with_propensity_scores(
    X: np.array, y: np.array, T: np.array, model_type: str = "LogisticRegression"
):
    # define the model for computing propensity scores
    # TODO: allow other model types
    clf = LogisticRegression()
    clf.fit(X, T)

    # compute propensity scores
    t_one_selector = T == 1
    t_zero_selector = T == 0
    propensity_scores = clf.predict_proba(X)[:, 1]

    # Naive ATE estimator
    p_t_one = clf.predict_proba(X[t_one_selector])[:, 1]
    p_t_zero = clf.predict_proba(X[t_zero_selector])[:, 0]
    num_units = len(X)
    ATE = (y[t_one_selector] / p_t_one).sum() / num_units - (
        y[t_zero_selector] / p_t_zero
    ).sum() / num_units

    # TODO: implement Hajek estimator

    return ATE, propensity_scores
