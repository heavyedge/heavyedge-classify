import numpy as np
import pytest
from heavyedge import ProfileData

from heavyedge_classify.api import classify_train
from heavyedge_classify.calibration import (
    TemperatureCalibratedClassifierCV,
    _SKLEARN_GE_1_8,
)
from heavyedge_classify.model import minirocket_classifier


def test_calibration_methods(tmp_traindata_path):
    profile_path, label_npy_path, _ = tmp_traindata_path

    X, _, _ = ProfileData(profile_path)[:]
    y = np.load(label_npy_path)

    for calibration in [
        "sigmoid",
        "isotonic",
        "temperature",
        "sigmoid_ovo",
        "isotonic_ovo",
    ]:
        model = minirocket_classifier(
            cv=2,
            calibration=calibration,
            random_state=42,
        )
        model.fit(X, y)


def test_classify_train(tmp_traindata_path):
    profile_path, label_npy_path, _ = tmp_traindata_path

    for calibration in [
        "sigmoid",
        "isotonic",
        "temperature",
        "sigmoid_ovo",
        "isotonic_ovo",
    ]:
        profiles = ProfileData(profile_path)
        labels = np.load(label_npy_path)
        classify_train(
            profiles,
            labels,
            cv=2,
            calibration=calibration,
            random_state=42,
        )


def test_temperature_calibration_alias():
    if _SKLEARN_GE_1_8:
        model = TemperatureCalibratedClassifierCV(
            estimator="estimator", cv=2, n_jobs=3
        )
    else:
        with pytest.deprecated_call(match="TemperatureCalibratedClassifierCV"):
            model = TemperatureCalibratedClassifierCV(
                estimator="estimator", cv=2, n_jobs=3
            )

    assert isinstance(model, TemperatureCalibratedClassifierCV)
    assert model.estimator == "estimator"
    assert model.cv == 2
    assert model.n_jobs == 3
