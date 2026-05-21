"""MiniRocket-based probabilistic classifier of 1D signals."""

from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .calibration import (
    IsotonicOvOCalibratedClassifierCV,
    SigmoidOvOCalibratedClassifierCV,
    TemperatureCalibratedClassifierCV,
)

__all__ = [
    "minirocket_classifier",
]


def minirocket_classifier(
    n_splits, calibration="sigmoid", n_jobs=None, verbose=False, random_state=0
):
    """MiniRocket-based probabilistic classifier of 1D signals.

    Parameters
    ----------
    n_splits : int
        Number of splits for cross-validation.
    calibration : {"sigmoid", "isotonic", "temperature", "sigmoid_ovo", "isotonic_ovo"}
        Calibration method for the classifier.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : bool, default=False
        Prints pipeline steps.
    random_state : int, default=0
        Random seed for reproducibility.

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
    >>> model = minirocket_classifier(n_splits=5, random_state=42)
    >>> X, _, _ = ProfileData(get_sample_path("Profiles.h5"))[:]
    >>> y = np.load(get_sample_path("labels.npy"))
    >>> model.fit(X[:5], y[:5])
    CalibratedClassifierCV(...)
    """
    pipeline = Pipeline(
        [
            ("minirocket", MiniRocket(random_state=random_state)),
            ("classifier", RidgeClassifierCV(class_weight="balanced")),
        ],
        verbose=verbose,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if calibration in ("sigmoid", "isotonic"):
        model = CalibratedClassifierCV(
            estimator=pipeline,
            method=calibration,
            cv=cv,
            n_jobs=n_jobs,
        )
    elif calibration == "temperature":
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
