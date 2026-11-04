"""3편 실습 코드(practice/ch03)가 합성 데이터와 임시 로컬 MLflow 저장소로 끝까지 도는지 확인한다.

모델 기록은 패키지 추론 때문에 한 번에 몇 초씩 걸리므로, 학습은 모듈에서 한 번만 하고 나눠 쓴다.
"""

import importlib
import math
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from mlops.slices import read_slice, write_slices
from mlops.tasks.timeseries import (
    CURRENT_OT,
    LOAD_COLUMNS,
    TASK,
    build_features,
    split_by_quarter,
)

PRACTICE_DIR = Path(__file__).resolve().parents[2] / "practice" / "ch03"
# 2016Q3부터 2017Q3(predict.py의 기본 슬라이스)까지 1시간 간격
SYNTHETIC_START = "2016-07-01"
SYNTHETIC_END = "2017-10-01"
FAST = ["--param", "n_estimators=30", "--param", "early_stopping_rounds=5"]
METRIC_KEYS = {"mae", "rmse", "naive_mae", "skill", "rows"}
MODEL_NAME = "test-etth1"


def make_raw() -> pd.DataFrame:
    dates = pd.date_range(SYNTHETIC_START, SYNTHETIC_END, freq="h", inclusive="left")
    steps = np.arange(len(dates), dtype=float)
    daily = np.sin(2 * np.pi * dates.hour / 24)
    loads = {column: daily * (i + 1) for i, column in enumerate(LOAD_COLUMNS)}
    return pd.DataFrame({"date": dates, **loads, "OT": 20 + 5 * daily + steps * 1e-3})


class ConstantChange:
    """변화량을 늘 같은 값으로 예측하는 가짜 모델."""

    def __init__(self, change: float) -> None:
        self.change = change

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.change)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """합성 데이터를 만들고, 임시 폴더를 로컬 MLflow 저장소로 삼아 run 3개를 쌓는다.

    run.sh처럼 MLFLOW_TRACKING_URI 없이 지금 폴더의 mlflow.db에 기록되게 한다.
    """
    root = tmp_path_factory.mktemp("ch03")
    data_dir = root / "data"
    write_slices(data_dir, TASK.name, split_by_quarter(build_features(make_raw())), TASK.source)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        mp.delenv("MLFLOW_TRACKING_URI", raising=False)
        # register.py가 같은 폴더의 train.py를 import한다
        mp.syspath_prepend(str(PRACTICE_DIR))
        train = importlib.import_module("train")
        data = ["--data-dir", str(data_dir)]
        metrics = {
            lr: train.main([*FAST, "--param", f"learning_rate={lr}", *data]) for lr in (0.05, 0.3)
        }
        train.main([*FAST, "--autolog", *data])
        yield {"root": root, "data_dir": data_dir, "data": data, "metrics": metrics}


def runs_by_logging(kind: str) -> list:
    return mlflow.search_runs(
        experiment_names=["etth1-ot-6h"],
        filter_string=f"tags.logging = '{kind}'",
        output_format="list",
    )


def test_local_store_is_a_sqlite_file_in_the_working_directory(store):
    assert mlflow.get_tracking_uri() == f"sqlite:///{store['root'] / 'mlflow.db'}"
    assert (store["root"] / "mlruns").is_dir()


def test_manual_run_records_params_metrics_curve_inputs_and_model(store):
    runs = runs_by_logging("manual")
    run = next(r for r in runs if r.data.params["learning_rate"] == "0.05")

    assert METRIC_KEYS <= set(run.data.metrics)
    assert all(math.isfinite(run.data.metrics[key]) for key in METRIC_KEYS)
    assert run.data.metrics["mae"] == pytest.approx(store["metrics"][0.05]["mae"])
    assert run.data.params["train_slices"] == ",".join(TASK.train_slices)
    assert {d.dataset.name for d in run.inputs.dataset_inputs} == {
        "etth1-2016Q3+2016Q4+2017Q1",
        "etth1-2017Q2",
    }
    curve = MlflowClient().get_metric_history(run.info.run_id, "valid_l1_by_tree")
    assert sorted(m.step for m in curve) == list(range(len(curve)))
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format="list",
    )
    assert len(models) == 1


def test_autolog_run_has_lightgbm_params_but_not_task_metrics(store):
    (run,) = runs_by_logging("autolog")

    assert "learning_rate" in run.data.params
    assert "mae" not in run.data.metrics


def test_register_alias_and_load_follow_the_alias(store, capsys):
    register = importlib.import_module("register")
    predict = importlib.import_module("predict")
    best, other = sorted(runs_by_logging("manual"), key=lambda r: r.data.metrics["mae"])
    name = ["--model-name", MODEL_NAME]

    register.main([*name, "register"])
    register.main([*name, "register", "--run-id", other.info.run_id])
    register.main([*name, "alias", "1"])
    first = predict.load_by_alias(MODEL_NAME, "champion")
    register.main([*name, "alias", "2"])
    second = predict.load_by_alias(MODEL_NAME, "champion")
    register.main([*name, "list"])

    assert first.metadata.run_id == best.info.run_id
    assert second.metadata.run_id == other.info.run_id
    assert "champion: 버전 1 → 버전 2" in capsys.readouterr().out


def test_predict_reports_mae_of_the_champion(store, capsys):
    predict = importlib.import_module("predict")
    register = importlib.import_module("register")
    register.main(["--model-name", "test-predict", "register"])
    register.main(["--model-name", "test-predict", "alias", "1"])

    mae = predict.main(["--model-name", "test-predict", *store["data"], "--rows", "2"])

    assert math.isfinite(mae)
    assert "2017Q3 전체" in capsys.readouterr().out


def test_register_explains_when_there_is_no_run(store):
    register = importlib.import_module("register")

    with pytest.raises(SystemExit, match="manual run이 없다"):
        register.main(["register", "--experiment", "no-such-experiment"])


def test_register_rejects_names_that_would_break_the_search_filter(store):
    register = importlib.import_module("register")

    with pytest.raises(SystemExit, match="이름에는"):
        register.main(["--model-name", "x' or '1'='1", "list"])


def test_inspect_store_shows_tables_and_model_files(store, capsys):
    inspect_store = importlib.import_module("inspect_store")

    inspect_store.main()

    out = capsys.readouterr().out
    assert "logged_models" in out
    assert "model_versions" in out
    assert "MLmodel" in out


def test_forecast_adds_current_temperature_to_predicted_change(store):
    predict = importlib.import_module("predict")
    df = read_slice(store["data_dir"], TASK.name, "2017Q3")

    forecast = predict.forecast(ConstantChange(1.5), df)

    assert np.allclose(forecast, df[CURRENT_OT].to_numpy() + 1.5)
