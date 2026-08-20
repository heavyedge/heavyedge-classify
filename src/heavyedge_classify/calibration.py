"""OvO calibration methods for scikit-learn."""

from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "SigmoidOvOCalibratedClassifierCV",
    "IsotonicOvOCalibratedClassifierCV",
]


def _get_response(estimator, X):
    """Get decision_function or predict_proba from estimator."""
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict_proba(X)


def _ovo_couple(r_pairs, n_classes, n_samples):
    """Convert OvO pairwise probabilities to multiclass probabilities.

    Uses normalised voting: ``p_i ∝ Σ_{j≠i} r_ij``.
    """
    votes = np.zeros((n_samples, n_classes))
    for (i, j), p_ij in r_pairs.items():
        votes[:, i] += p_ij
        votes[:, j] += 1.0 - p_ij
    total = votes.sum(axis=1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return votes / total


class _OvOCalibratedClassifier:
    """A fitted estimator paired with OvO pairwise calibrators."""

    def __init__(self, estimator, calibrators, classes):
        self.estimator = estimator
        self.calibrators = calibrators  # dict of (i, j) -> calibrator
        self.classes = classes

    def predict_proba(self, X):
        raw = _get_response(self.estimator, X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        r_pairs = {}
        for (i, j), cal in self.calibrators.items():
            if raw.ndim == 1:
                f = raw
            else:
                f = raw[:, i] - raw[:, j]
            r_pairs[(i, j)] = cal.predict(f)

        proba = _ovo_couple(r_pairs, n_classes, n_samples)
        proba = np.clip(proba, 0.0, 1.0)
        return proba


def _fit_one_fold_ovo(estimator, X, y, train, test, classes, method):
    """Clone *estimator*, fit on *train*, calibrate OvO on *test*."""
    est = clone(estimator)
    est.fit(X[train], y[train])

    raw = _get_response(est, X[test])

    le = LabelEncoder().fit(classes)
    y_enc = le.transform(y[test])

    n_classes = len(classes)
    calibrators = {}

    for i, j in combinations(range(n_classes), 2):
        mask = (y_enc == i) | (y_enc == j)
        if mask.sum() == 0:
            continue

        if raw.ndim == 1:
            f_pair = raw[mask]
        else:
            f_pair = raw[mask, i] - raw[mask, j]

        y_binary = (y_enc[mask] == i).astype(np.float64)

        if method == "sigmoid":
            cal = _SigmoidCalibration()
            cal.fit(f_pair, y_binary)
        else:  # isotonic
            cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            cal.fit(f_pair, y_binary)

        calibrators[(i, j)] = cal

    return _OvOCalibratedClassifier(est, calibrators, classes)


class _OvOCalibratedClassifierCV(ClassifierMixin, BaseEstimator):
    """Base class for OvO-calibrated classifiers with cross-validation.

    Each CV fold produces an independent (estimator, pairwise-calibrators)
    pair.  Predictions are averaged at inference time (``ensemble=True``).
    """

    _method = None  # override in subclasses

    def __init__(self, estimator, *, cv=5, n_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        le = LabelEncoder().fit(y)
        self.classes_ = le.classes_

        if hasattr(self.cv, "split"):
            splits = list(self.cv.split(X, y))
        else:
            splits = list(self.cv)

        self.calibrated_classifiers_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_fold_ovo)(
                self.estimator, X, y, train, test, self.classes_, self._method
            )
            for train, test in splits
        )

        first = self.calibrated_classifiers_[0].estimator
        if hasattr(first, "n_features_in_"):
            self.n_features_in_ = first.n_features_in_
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for cc in self.calibrated_classifiers_:
            mean_proba += cc.predict_proba(X)
        mean_proba /= len(self.calibrated_classifiers_)
        return mean_proba

    def predict(self, X):
        check_is_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class SigmoidOvOCalibratedClassifierCV(_OvOCalibratedClassifierCV):
    """Cross-validated one-versus-one calibration."""

    _method = "sigmoid"


class IsotonicOvOCalibratedClassifierCV(_OvOCalibratedClassifierCV):
    """Cross-validated isotonic one-versus-one calibration."""

    _method = "isotonic"
