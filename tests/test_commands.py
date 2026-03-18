import os
import subprocess


def test_train_log(tmp_traindata_path, tmp_path):
    profile_path, label_path = tmp_traindata_path
    model_path = tmp_path / "model.pkl"
    result = subprocess.run(
        [
            "heavyedge",
            "--log-level=INFO",
            "classify-train",
            profile_path,
            label_path,
            "-o",
            model_path,
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert os.path.exists(model_path)
    assert "[Pipeline]" in result.stderr
