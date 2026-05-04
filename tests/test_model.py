import numpy as np
from heavyedge import ProfileData

from heavyedge_classify.api import classify_train
from heavyedge_classify.model import minirocket_classifier


def test_calibration_methods(tmp_traindata_path):
    profile_path, label_npy_path, _ = tmp_traindata_path

    X, _, _ = ProfileData(profile_path)[:]
    y = np.load(label_npy_path)

    for calibration in ["sigmoid", "isotonic"]:
        model = minirocket_classifier(
            n_splits=2,
            calibration=calibration,
            random_state=42,
        )
        model.fit(X, y)


def test_classify_train(tmp_traindata_path):
    profile_path, label_npy_path, _ = tmp_traindata_path

    for calibration in ["sigmoid", "isotonic"]:
        profiles = ProfileData(profile_path)
        labels = np.load(label_npy_path)
        classify_train(
            profiles,
            labels,
            n_splits=2,
            calibration=calibration,
            random_state=42,
        )
