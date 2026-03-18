"""High-level Python runtime interface."""

import contextlib
import io
import sys

import numpy as np

from ..model import minirocket_classifier

__all__ = [
    "classify_train",
    "classify_predict",
]


class _LoggerStream(io.TextIOBase):
    """Stream that forwards written lines to a logger callback."""

    def __init__(self, logger, original_stdout):
        self.logger = logger
        self._original_stdout = original_stdout
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                sys.stdout = self._original_stdout
                try:
                    self.logger(line)
                finally:
                    sys.stdout = self
        return len(s)

    def flush(self):
        if self._buf.strip():
            sys.stdout = self._original_stdout
            try:
                self.logger(self._buf.strip())
            finally:
                sys.stdout = self
        self._buf = ""


def classify_train(
    profiles, labels, n_splits=5, normalize=True, random_state=0, logger=lambda x: None
):
    """Train classification model.

    Parameters
    ----------
    profiles : heavyedge.ProfileData
        Open h5 file of profiles.
    labels : np.ndarray
        Label array. The order of labels should match the order of profiles.
    n_splits : int, default=5
        Number of splits for cross-validation.
    normalize : bool, default=True
        Whether to normalize profiles by area under curve.
        Set this to False if *profiles* are already normalized.
    random_state : int, default=0
        Random seed for reproducibility.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Returns
    -------
    model
        Trained model object.

    Examples
    --------
    >>> from heavyedge import ProfileData
    >>> from heavyedge_classify.samples import get_sample_path
    >>> from heavyedge_classify.api import classify_train
    >>> import numpy as np
    >>> profiles = ProfileData(get_sample_path("Profiles.h5"))
    >>> labels = np.load(get_sample_path("labels.npy"))
    >>> classify_train(profiles, labels)
    CalibratedClassifierCV(...)
    """
    x = profiles.x()
    X, _, _ = profiles[:]
    if normalize:
        X /= np.trapezoid(X, x, axis=1)[..., np.newaxis]
    model = minirocket_classifier(
        n_splits=n_splits, verbose=True, random_state=random_state
    )
    with contextlib.redirect_stdout(_LoggerStream(logger, sys.stdout)):
        model.fit(X, labels)
    return model


def classify_predict(
    model, profiles, normalize=True, batch_size=None, logger=lambda x: None
):
    """Predict probabilistic labels of profiles using a trained model.

    Parameters
    ----------
    model
        Trained model object.
    profiles : heavyedge.ProfileData
        Open h5 file of profiles.
    normalize : bool, default=True
        Whether to normalize profiles by area under curve.
        Set this to False if *profiles* are already normalized.
    batch_size : int, optional
        Batch size to load data.
        If not passed, all data are loaded at once.
    logger : callable, optional
        Logger function which accepts a progress message string.

    Yields
    ------
    predicted_labels : np.ndarray
        Predicted probabilistic label array.

    Examples
    --------
    >>> import pickle
    >>> from heavyedge import ProfileData
    >>> from heavyedge_classify.samples import get_sample_path
    >>> from heavyedge_classify.api import classify_predict
    >>> with open(get_sample_path("model.pkl"), "rb") as f:
    ...     model = pickle.load(f)
    >>> profiles = ProfileData(get_sample_path("Profiles.h5"))
    >>> [lab.shape for lab in classify_predict(model, profiles, batch_size=50)]
    [(50, 3), (25, 3)]
    """
    x = profiles.x()
    N, _ = profiles.shape()

    if batch_size is None:
        X, _, _ = profiles[:]
        if normalize:
            X /= np.trapezoid(X, x, axis=1)[..., np.newaxis]
        yield model.predict_proba(X)
        logger(f"{N}/{N}")
    else:
        for i in range(0, N, batch_size):
            X, _, _ = profiles[i : i + batch_size]
            if normalize:
                X /= np.trapezoid(X, x, axis=1)[..., np.newaxis]
            yield model.predict_proba(X)
            logger(f"{min(i + batch_size, N)}/{N}")
