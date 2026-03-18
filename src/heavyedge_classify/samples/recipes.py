"""Recipies to build sample data."""

import numpy as np
from heavyedge import ProfileData
from heavyedge import get_sample_path as heavyedge_sample

from . import get_sample_path


def save_labels(path):
    labels = []
    with ProfileData(heavyedge_sample("Prep-Type1.h5")) as data:
        labels += [1 for _ in range(data.shape()[0])]
    with ProfileData(heavyedge_sample("Prep-Type2.h5")) as data:
        labels += [2 for _ in range(data.shape()[0])]
    with ProfileData(heavyedge_sample("Prep-Type3.h5")) as data:
        labels += [3 for _ in range(data.shape()[0])]
    np.save(path, labels)


RECIPES = {
    "Profiles.h5": lambda path: [
        "heavyedge",
        "merge",
        heavyedge_sample("Prep-Type1.h5"),
        heavyedge_sample("Prep-Type2.h5"),
        heavyedge_sample("Prep-Type3.h5"),
        "-o",
        path,
    ],
    "labels.npy": lambda path: save_labels(path),
    "model.pkl": lambda path: [
        "heavyedge",
        "classify-train",
        get_sample_path("Profiles.h5"),
        get_sample_path("labels.npy"),
        "-o",
        path,
    ],
}
