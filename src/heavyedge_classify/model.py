"""MiniRocket-based probabilistic classifier of 1D signals."""

import warnings

from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn import __version__ as _sklearn_version
from packaging.version import Version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .calibration import (
    IsotonicOvOCalibratedClassifierCV,
    SigmoidOvOCalibratedClassifierCV,
    TemperatureCalibratedClassifierCV,
)

_SKLEARN_GE_1_8 = Version(_sklearn_version) >= Version("1.8.0")

__all__ = [
    "minirocket_classifier",
]


def minirocket_classifier(
    cv=5,
    calibration="sigmoid",
    n_jobs=None,
    verbose=False,
    random_state=0,
    n_splits=None,
):
    """MiniRocket-based probabilistic classifier of 1D signals.

    Parameters
    ----------
    cv : int, iterable, or cross-validation generator, default=5
        Cross-validation strategy.
        If an integer is passed, it is the number of folds for stratified k-fold CV.
    calibration : {"sigmoid", "isotonic", "temperature", "sigmoid_ovo", "isotonic_ovo"}
        Calibration method for the classifier.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : bool, default=False
        Prints pipeline steps.
    random_state : int, default=0
        Random seed for reproducibility.
    n_splits : int, optional
        Number of splits for cross-validation.
        If passed, overrides *cv*.

        .. deprecated:: 1.4.0
            The *n_splits* parameter is deprecated and will be removed in a future
            version. Use *cv* instead.

    Returns
    -------
    model
        MiniRocket-based probabilistic classifier.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_classify.samples import get_sample_path
    >>> from heavyedge_classify.model import minirocket_classifier
    >>> import numpy as np
    >>> model = minirocket_classifier(cv=5, random_state=42)
    >>> X, _, _ = ProfileData(get_sample_path("Profiles.h5"))[:]
    >>> y = np.load(get_sample_path("labels.npy"))
    >>> model.fit(X[:5], y[:5])
    CalibratedClassifierCV(...)
    """
    if n_splits is not None:
        warnings.warn(
            (
                "n_splits is deprecated and will be removed in a future version. "
                "Use cv instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        cv = n_splits
    pipeline = Pipeline(
        [
            ("minirocket", MiniRocket(random_state=random_state)),
            ("classifier", RidgeClassifierCV(class_weight="balanced")),
        ],
        verbose=verbose,
    )
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    if calibration in ("sigmoid", "isotonic"):
        model = CalibratedClassifierCV(
            estimator=pipeline,
            method=calibration,
            cv=cv,
            n_jobs=n_jobs,
        )
    elif calibration == "temperature":
        if _SKLEARN_GE_1_8:
            model = CalibratedClassifierCV(
                estimator=pipeline,
                method="temperature",
                cv=cv,
                n_jobs=n_jobs,
            )
        else:
            model = TemperatureCalibratedClassifierCV(
                estimator=pipeline,
                cv=cv,
                n_jobs=n_jobs,
            )
    elif calibration == "sigmoid_ovo":
        model = SigmoidOvOCalibratedClassifierCV(
            estimator=pipeline,
            cv=cv,
            n_jobs=n_jobs,
        )
    elif calibration == "isotonic_ovo":
        model = IsotonicOvOCalibratedClassifierCV(
            estimator=pipeline,
            cv=cv,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unsupported calibration method: {calibration}")
    return model
