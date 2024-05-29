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
    naive_ATE = (y[t_one_selector] / p_t_one).sum() / num_units - (
        y[t_zero_selector] / p_t_zero
    ).sum() / num_units

    # Hajek estimator
    prob_predictions = clf.predict_proba(X)
    hajek_weights = (t_one_selector / prob_predictions[:, 1]) + (
        (1 - t_zero_selector) / (1 - prob_predictions[:, 1])
    )
    hw_t_one = hajek_weights[t_one_selector]
    hw_t_zero = hajek_weights[t_zero_selector]

    hajek_ATE = (y[t_one_selector].dot(hw_t_one) / np.sum(hw_t_one)) - (
        y[t_zero_selector].dot(hw_t_zero) / np.sum(hw_t_zero)
    )

    return hajek_ATE, naive_ATE, propensity_scores
