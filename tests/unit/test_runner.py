import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from mlops.runner import (
    ExportMismatchError,
    compare_predictions,
    parse_params,
    run_prepare,
    run_training,
    sample_rows,
)
from mlops.slices import read_manifest
from tests.fakes import MeanTask

DEFAULTS = {"lr": 0.1, "epochs": 2, "name": "a", "flag": False, "limit": None}


def test_parse_params_casts_values_to_the_default_types():
    params = parse_params(DEFAULTS, ["lr=0.5", "epochs=3", "name=b", "flag=true", "limit=10"])

    assert params == {"lr": 0.5, "epochs": 3, "name": "b", "flag": True, "limit": 10}
    assert isinstance(params["lr"], float)


def test_parse_params_keeps_defaults_and_does_not_mutate_them():
    params = parse_params(DEFAULTS, ["epochs=5"])

    assert params["lr"] == 0.1
    assert DEFAULTS["epochs"] == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [("unknown=1", "알 수 없는 파라미터"), ("lr", "key=value"), ("epochs=abc", "epochs")],
)
def test_parse_params_rejects_bad_input(override, message):
    with pytest.raises(ValueError, match=message):
        parse_params(DEFAULTS, [override])


def test_compare_predictions_accepts_float_noise_within_tolerance():
    result = compare_predictions(np.array([10.0, 20.0]), np.array([10.00001, 19.99999]))

    assert result["kind"] == "float"
    assert result["passed"]


def test_compare_predictions_flags_float_differences():
    result = compare_predictions(np.array([10.0, 20.0]), np.array([10.0, 21.0]))

    assert not result["passed"]
    assert result["max_abs_diff"] == pytest.approx(1.0)


def test_compare_predictions_measures_label_agreement():
    result = compare_predictions(np.array(["a", "b", "c", "d"]), np.array(["a", "b", "c", "x"]))

    assert result["kind"] == "label"
    assert result["agreement"] == pytest.approx(0.75)
    assert not result["passed"]


def test_sample_rows_is_deterministic_and_bounded():
    df = pd.DataFrame({"x": range(100)})

    first = sample_rows(df, 10)

    assert len(first) == 10
    assert first["x"].tolist() == sample_rows(df, 10)["x"].tolist()
    assert len(sample_rows(df, None)) == 100
    assert len(sample_rows(df, 1000)) == 100


def test_run_prepare_writes_every_slice_with_manifest(tmp_path):
    run_prepare(MeanTask(), tmp_path)

    manifest = read_manifest(tmp_path, "mean")
    assert [s["name"] for s in manifest["slices"]] == ["s1", "s2", "s3"]
    assert manifest["source"] == {"kind": "synthetic"}


def test_run_prepare_rejects_tasks_that_return_unexpected_slices(tmp_path):
    @dataclass(frozen=True)
    class MissingSlice(MeanTask):
        def prepare(self):
            return {"s1": pd.DataFrame({"y": [1.0]})}

    with pytest.raises(ValueError, match="s2"):
        run_prepare(MissingSlice(), tmp_path)


def test_run_training_evaluates_every_slice_and_writes_report(tmp_path):
    task = MeanTask()
    run_prepare(task, tmp_path / "data")

    report = run_training(task, tmp_path / "data", tmp_path / "out", params={"shift": 0.0})

    assert report["metrics"]["s1"]["mae"] == pytest.approx(2 / 3)
    assert report["metrics"]["s3"]["mae"] == pytest.approx(8.5)
    assert report["export_check"]["passed"]
    assert report["params"] == {"shift": 0.0}
    assert json.loads((tmp_path / "out" / "report.json").read_text()) == report


def test_run_training_fails_when_exported_model_disagrees(tmp_path):
    task = MeanTask(broken_export=True)
    run_prepare(task, tmp_path / "data")

    with pytest.raises(ExportMismatchError):
        run_training(task, tmp_path / "data", tmp_path / "out", params={"shift": 0.0})

    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert not report["export_check"]["passed"]


def test_compare_predictions_reports_nan_as_failure_with_valid_json():
    result = compare_predictions(np.array([1.0, 2.0]), np.array([1.0, np.nan]))

    assert not result["passed"]
    assert result["max_abs_diff"] is None
    json.dumps(result, allow_nan=False)


def test_rerun_replaces_previous_results(tmp_path):
    task = MeanTask()
    run_prepare(task, tmp_path / "data")
    run_training(task, tmp_path / "data", tmp_path / "out", params={"shift": 0.0})
    (tmp_path / "out" / "stale.txt").write_text("이전 실행의 파일")

    report = run_training(task, tmp_path / "data", tmp_path / "out", params={"shift": 1.0})

    assert not (tmp_path / "out" / "stale.txt").exists()
    assert report["params"] == {"shift": 1.0}


def test_failed_training_keeps_the_previous_results(tmp_path):
    @dataclass(frozen=True)
    class CrashingTask(MeanTask):
        def train(self, train, valid, params):
            raise RuntimeError("학습 중 중단")

    run_prepare(MeanTask(), tmp_path / "data")
    run_training(MeanTask(), tmp_path / "data", tmp_path / "out", params={"shift": 0.0})

    with pytest.raises(RuntimeError, match="중단"):
        run_training(CrashingTask(), tmp_path / "data", tmp_path / "out", params={"shift": 2.0})

    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["params"] == {"shift": 0.0}


def test_run_training_refuses_to_replace_an_unrelated_folder(tmp_path):
    run_prepare(MeanTask(), tmp_path / "data")
    out = tmp_path / "notes"
    out.mkdir()
    (out / "memo.txt").write_text("지우면 안 되는 파일")

    with pytest.raises(ValueError, match="학습 결과가 아닌"):
        run_training(MeanTask(), tmp_path / "data", out)

    assert (out / "memo.txt").exists()


def test_slices_can_be_moved_forward_for_retraining(tmp_path):
    task = MeanTask()
    run_prepare(task, tmp_path / "data")

    report = run_training(
        task, tmp_path / "data", tmp_path / "out", train_slices=["s1", "s2"], valid_slices=["s3"]
    )

    assert report["train_slices"] == ["s1", "s2"]
    assert report["valid_slices"] == ["s3"]
    assert report["metrics"]["s3"]["mae"] == pytest.approx(10.5 - 2.2)


@pytest.mark.parametrize(
    ("train", "valid", "message"),
    [(["s1"], ["s9"], "없는 슬라이스"), (["s1", "s2"], ["s2"], "겹친다")],
)
def test_slice_override_is_validated(tmp_path, train, valid, message):
    run_prepare(MeanTask(), tmp_path / "data")

    with pytest.raises(ValueError, match=message):
        run_training(
            MeanTask(), tmp_path / "data", tmp_path / "out", train_slices=train, valid_slices=valid
        )
