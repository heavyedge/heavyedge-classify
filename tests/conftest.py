import csv
import subprocess

import numpy as np
import pytest

np.random.seed(0)


def profile_type2(b0, b1, b2, b3, data_size=None, random_scale=0):
    """Generate artificial Type 2 profile.

    Parameters
    ----------
    b0 : scalar
        Plateau region height.
    b1, b2, b3 : scalar
        Heavy edge region y = -b1 * (x - b2)^2 + b3.
    data_size : int, optional
        If passed, profile is padded with "bare substrate region" to this size.
    random_scale : scalar, default=0
        Scale for standard normal noise.

    Raises
    ------
    ValueError
        If profile lenth is larger than *data_size*.
    """
    x = np.arange(np.ceil(np.sqrt(b3 / b1) + b2).astype(int))
    x_bp = np.ceil(-np.sqrt((b3 - b0) / b1) + b2).astype(int)

    y1 = np.full(x.shape, b0)
    y1[x_bp:] = 0
    y2 = -b1 * ((x - b2) ** 2) + b3
    y2[:x_bp] = 0
    ret = (y1 + y2).astype(float)

    if data_size is not None:
        if len(ret) > data_size:
            raise ValueError("data_size is too small.")
        ret = np.pad(ret, (0, data_size - len(ret)))
    ret += np.random.standard_normal(ret.shape) * random_scale
    return ret


class RawDataFactory:
    def __init__(self, path):
        self.path = path

    def mkrawdir(self, dirname):
        path = self.path / dirname
        path.mkdir(parents=True)
        return path

    def mkrawfile(self, rawdir, filename, data):
        path = rawdir / filename
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for y in data:
                writer.writerow([y])
        return path


@pytest.fixture(scope="session")
def tmp_traindata_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("RawData-")
    rawdata_factory = RawDataFactory(path)

    N_PROFILES = 5
    DATA_SIZE = 150
    RANDOM_SCALE = 5

    rawdir = rawdata_factory.mkrawdir("Type2-00")
    for i in range(N_PROFILES):
        rawdata_factory.mkrawfile(
            rawdir,
            f"{str(i).zfill(2)}.csv",
            profile_type2(
                700, 1, 50, 800, data_size=DATA_SIZE, random_scale=RANDOM_SCALE
            ),
        )

    profile_path = tmp_path_factory.mktemp("PrepData-") / "Type2.h5"
    subprocess.run(
        [
            "heavyedge",
            "prep",
            "--type",
            "csvs",
            "--res=1",
            "--sigma=1",
            "--std-thres=40",
            "--fill-value=0",
            "--z-thres=3.5",
            rawdir,
            "-o",
            profile_path,
        ],
        capture_output=True,
        check=True,
    )
    label_path = tmp_path_factory.mktemp("Label-") / "labels.npy"
    np.save(label_path, [1] * N_PROFILES)

    return (profile_path, label_path)
