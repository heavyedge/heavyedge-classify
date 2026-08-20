"""Calibration methods for scikit-learn."""

import warnings
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from packaging.version import Version
from scipy.optimize import minimize_scalar
from sklearn import __version__ as _sklearn_version
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted

_SKLEARN_GE_1_8 = Version(_sklearn_version) >= Version("1.8.0")

__all__ = [
    "TemperatureCalibratedClassifierCV",
    "SigmoidOvOCalibratedClassifierCV",
    "IsotonicOvOCalibratedClassifierCV",
]


def _get_response(estimator, X):
    """Get decision_function or predict_proba from estimator."""
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict_proba(X)


def _convert_to_logits(values, eps=1e-12):
    """Convert decision_function / predict_proba output to 2-D logits.

    - 1-D decision values ``(n,)`` -> ``(n, 2)`` as ``(-x, x)``.
    - 2-D with one column ``(n, 1)`` -> same treatment.
    - 2-D probabilities (rows sum to 1) -> ``log(p + eps)``.
    - Otherwise returned as-is.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 2 and values.shape[1] > 1:
        is_prob = np.all(values >= 0) and np.all(values <= 1)
        sums_one = np.allclose(values.sum(axis=1), 1.0)
        if is_prob and sums_one:
            return np.log(values + eps)
        return values

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    # binary: shape (n, 1) -> (n, 2)
    return np.concatenate([-values, values], axis=1)


class _TemperatureScaling:
    """Learn a single inverse-temperature beta so that ``softmax(beta * logits)``
    minimises the multinomial cross-entropy on a calibration set.
    """

    def fit(self, X, y, sample_weight=None):
        logits = _convert_to_logits(X)
        dtype = logits.dtype
        labels = np.asarray(y, dtype=np.intp).ravel()

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=dtype)
        else:
            sw = None

        n = len(labels)
        idx = np.arange(n)

        def _nll(log_beta):
            beta = np.exp(log_beta)
            probs = softmax(beta * logits)
            lp = np.log(np.clip(probs[idx, labels], 1e-15, None))
            if sw is not None:
                return -(sw * lp).sum() / sw.sum()
            return -lp.mean()

        res = minimize_scalar(
            _nll,
            bounds=(-10.0, 10.0),
            method="bounded",
            options={"xatol": 64 * np.finfo(dtype).eps},
        )
        if not res.success:
            raise RuntimeError(
                "Temperature scaling optimisation failed: " + str(res.message)
            )

        self.beta_ = np.exp(res.x)
        return self

    def predict(self, X):
        logits = _convert_to_logits(X)
        return softmax(self.beta_ * logits)


class _CalibratedClassifier:
    """A fitted estimator paired with a fitted temperature calibrator."""

    def __init__(self, estimator, calibrator, classes):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes = classes

    def predict_proba(self, X):
        raw = _get_response(self.estimator, X)
        proba = self.calibrator.predict(raw)
        # Guard against tiny floating-point overshoots.
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0
        return proba


def _fit_one_fold(estimator, X, y, train, test, classes):
    """Clone *estimator*, fit on *train*, calibrate on *test*."""
    est = clone(estimator)
    est.fit(X[train], y[train])

    raw = _get_response(est, X[test])
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    le = LabelEncoder().fit(classes)
    y_enc = le.transform(y[test])

    cal = _TemperatureScaling()
    cal.fit(raw, y_enc)

    return _CalibratedClassifier(est, cal, classes)


class TemperatureCalibratedClassifierCV(ClassifierMixin, BaseEstimator):
    """Deprecated standalone cross-validated temperature-scaling calibration.

    .. deprecated::
        scikit-learn < 1.8.0 support is deprecated and will be removed in
        v2.0.0 of this package. Upgrade to scikit-learn >= 1.8.0 to use
        :class:`sklearn.calibration.CalibratedClassifierCV` with
        ``method='temperature'`` natively.
    """

    def __init__(self, estimator, *, cv=5, n_jobs=1):
        warnings.warn(
            (
                "scikit-learn < 1.8.0 support in TemperatureCalibratedClassifierCV "
                "is deprecated and will be removed in v2.0.0 of this package. "
                "Upgrade to scikit-learn >= 1.8.0."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
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
            delayed(_fit_one_fold)(self.estimator, X, y, train, test, self.classes_)
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


if _SKLEARN_GE_1_8:

    class TemperatureCalibratedClassifierCV(CalibratedClassifierCV):  # noqa: F811
        """Cross-validated temperature-scaling calibration.

        A thin subclass of :class:`sklearn.calibration.CalibratedClassifierCV`
        that defaults to ``method='temperature'`` (requires scikit-learn >= 1.8.0).
        """

        def __init__(self, estimator, *, cv=5, n_jobs=1):
            super().__init__(
                estimator=estimator,
                method="temperature",
                cv=cv,
                n_jobs=n_jobs,
            )

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
