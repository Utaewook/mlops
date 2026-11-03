import json
import logging

import pytest

from mlops import cli
from mlops.tasks import get_task
from tests.fakes import MeanTask


def test_unknown_task_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["train", "audio"])
    with pytest.raises(ValueError, match="audio"):
        get_task("audio")


def test_train_with_unknown_param_exits_with_error(tmp_path, caplog):
    code = cli.main(["train", "timeseries", "--data-dir", str(tmp_path), "--param", "nope=1"])

    assert code == 1
    assert "nope" in caplog.text


def test_train_without_prepared_data_explains_the_next_step(tmp_path, caplog):
    code = cli.main(["train", "timeseries", "--data-dir", str(tmp_path)])

    assert code == 1
    assert "python -m mlops prepare timeseries" in caplog.text


def test_prepare_then_train_writes_a_report(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(cli, "get_task", lambda name: MeanTask())
    data, out = str(tmp_path / "data"), tmp_path / "out"

    assert cli.main(["prepare", "timeseries", "--data-dir", data]) == 0
    code = cli.main(
        ["train", "timeseries", "--data-dir", data, "--out", str(out), "--param", "shift=1.5"]
    )

    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["params"] == {"shift": 1.5}
    assert "future" in caplog.text


def test_train_accepts_comma_separated_slices(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_task", lambda name: MeanTask())
    data, out = str(tmp_path / "data"), tmp_path / "out"
    cli.main(["prepare", "timeseries", "--data-dir", data])

    code = cli.main(
        [
            "train",
            "timeseries",
            "--data-dir",
            data,
            "--out",
            str(out),
            "--train-slices",
            "s1, s2",
            "--valid-slices",
            "s3",
        ]
    )

    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert (report["train_slices"], report["valid_slices"]) == (["s1", "s2"], ["s3"])
