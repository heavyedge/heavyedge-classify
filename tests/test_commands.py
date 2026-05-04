import csv
import os
import subprocess

import numpy as np


def test_train_label_npy(tmp_traindata_path, tmp_path):
    profile_path, label_npy_path, _ = tmp_traindata_path
    model_path = tmp_path / "model.pkl"
    subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "classify-train",
            profile_path,
            label_npy_path,
            "--n-splits=2",
            "-o",
            model_path,
        ],
        check=True,
    )
    assert os.path.exists(model_path)


def test_train_label_csv(tmp_traindata_path, tmp_path):
    profile_path, _, label_csv_path = tmp_traindata_path
    model_path = tmp_path / "model.pkl"
    subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "classify-train",
            profile_path,
            label_csv_path,
            "--n-splits=2",
            "-o",
            model_path,
        ],
        check=True,
    )
    assert os.path.exists(model_path)


def test_train_log(tmp_traindata_path, tmp_path):
    profile_path, label_npy_path, _ = tmp_traindata_path
    model_path = tmp_path / "model.pkl"
    result = subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "classify-train",
            profile_path,
            label_npy_path,
            "--n-splits=2",
            "-o",
            model_path,
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert os.path.exists(model_path)
    assert "[Pipeline]" in result.stderr


def test_predict_format_csv(tmp_traindata_path, tmp_model, tmp_path):
    profile_path, _, _ = tmp_traindata_path
    model_path = tmp_model

    # csv output by parsing from file extension
    output_path = tmp_path / "predictions.csv"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "-o",
            output_path,
        ],
        check=True,
    )
    with open(output_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) > 0
    # csv output by passing format
    output_path = tmp_path / "predictions.npy"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "--output-format",
            "csv",
            "-o",
            output_path,
        ],
        check=True,
    )
    with open(output_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) > 0


def test_predict_format_npy(tmp_traindata_path, tmp_model, tmp_path):
    profile_path, _, _ = tmp_traindata_path
    model_path = tmp_model

    # npy output by parsing from file extension
    output_path = tmp_path / "predictions.npy"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "-o",
            output_path,
        ],
        check=True,
    )
    data = np.load(output_path)
    assert data.shape[0] > 0
    # npy output by passing format
    output_path = tmp_path / "predictions.npy"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "--output-format",
            "npy",
            "-o",
            output_path,
        ],
        check=True,
    )
    data = np.load(output_path)
    assert data.shape[0] > 0


def test_predict_soft_label(tmp_traindata_path, tmp_model, tmp_path):
    profile_path, _, _ = tmp_traindata_path
    model_path = tmp_model

    output_path = tmp_path / "predictions.csv"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "--label-type",
            "soft",
            "-o",
            output_path,
        ],
        check=True,
    )

    output_path = tmp_path / "predictions.npy"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "--label-type",
            "soft",
            "-o",
            output_path,
        ],
        check=True,
    )


def test_predict_hard_label(tmp_traindata_path, tmp_model, tmp_path):
    profile_path, _, _ = tmp_traindata_path
    model_path = tmp_model

    output_path = tmp_path / "predictions.csv"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "--label-type",
            "hard",
            "-o",
            output_path,
        ],
        check=True,
    )

    output_path = tmp_path / "predictions.npy"
    subprocess.run(
        [
            "heavyedge",
            "classify-predict",
            profile_path,
            model_path,
            "--label-type",
            "hard",
            "-o",
            output_path,
        ],
        check=True,
    )


def test_calibration_methods(tmp_traindata_path, tmp_path):
    profile_path, label_npy_path, _ = tmp_traindata_path

    for calibration in ["sigmoid", "isotonic"]:
        model_path = tmp_path / f"model_{calibration}.pkl"
        subprocess.run(
            [
                "heavyedge",
                "--log-level=INFO",
                "classify-train",
                profile_path,
                label_npy_path,
                "--n-splits=2",
                "--calibration",
                calibration,
                "-o",
                model_path,
            ],
            check=True,
        )
        assert os.path.exists(model_path)
