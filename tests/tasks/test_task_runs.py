"""과제별 준비 → 학습 → 평가 → 내보내기를 실제 데이터로 한 번씩 돌린다.

데이터와 가중치를 내려받고(처음 한 번) 메모리를 쓰므로 과제마다 따로 돌린다.

    docker compose run --rm --no-deps workspace pytest -m timeseries
    docker compose run --rm --no-deps workspace pytest -m vision
    docker compose run --rm --no-deps workspace pytest -m llm
"""

import pytest

from mlops.runner import run_prepare, run_training
from mlops.slices import read_slice
from mlops.tasks import get_task

# LLM 테스트는 구조만 확인하므로 아주 작은 Qwen3 모델(무작위 가중치)을 쓴다
TINY_QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_QWEN3_REVISION = "25bb9b8963688237e94750f6f3e48f1947d3c7a3"


def prepare_and_train(tmp_path, name, params, max_eval_rows=None):
    task = get_task(name)
    run_prepare(task, tmp_path / "data")
    report = run_training(task, tmp_path / "data", tmp_path / "out", params, max_eval_rows)
    assert list(report["metrics"]) == list(task.slices)
    assert report["export_check"]["passed"]
    return report


@pytest.mark.timeseries
def test_timeseries_forecasts_beat_a_naive_baseline(tmp_path):
    report = prepare_and_train(tmp_path, "timeseries", {"n_estimators": 300})

    # 지금 온도가 HORIZON_HOURS 뒤에도 같다고 보는 예측(persistence)보다 나아야 한다
    valid = read_slice(tmp_path / "data", "timeseries", "2017Q2")
    naive_mae = (valid["target"] - valid["OT_lag0"]).abs().mean()
    assert (tmp_path / "out" / "model.onnx").exists()
    assert report["metrics"]["2017Q2"]["mae"] < naive_mae


@pytest.mark.vision
def test_vision_learns_and_degrades_under_haze(tmp_path):
    report = prepare_and_train(
        tmp_path, "vision", {"epochs": 1, "max_train_samples": 2048}, max_eval_rows=512
    )

    metrics = report["metrics"]
    assert (tmp_path / "out" / "preprocess.json").exists()
    # 2048장, 1 epoch만 학습해도 무작위 추측(0.1)보다 훨씬 낫다
    assert metrics["valid"]["accuracy"] > 0.4
    assert metrics["t3"]["accuracy"] < metrics["t0"]["accuracy"]


@pytest.mark.llm
def test_llm_trains_an_adapter_and_reloads_it(tmp_path):
    params = {
        "base_model": TINY_QWEN3,
        "base_revision": TINY_QWEN3_REVISION,
        "max_train_samples": 16,
        "batch_size": 4,
        "eval_batch_size": 4,
    }

    report = prepare_and_train(tmp_path, "llm", params, max_eval_rows=8)

    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()
    assert all(0.0 <= m["invalid_rate"] <= 1.0 for m in report["metrics"].values())
    assert report["data"]["slices"][-1]["name"] == "2020"
