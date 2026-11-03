"""과제와 상관없이 같은 순서로 도는 준비·학습 절차.

학습 → 슬라이스별 평가 → 내보내기 → 내보낸 파일로 다시 예측해서 학습 때와 같은지 확인 → report.json
"""

import gc
import json
import logging
import math
import resource
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops.slices import read_manifest, read_slice, read_slices, write_slices
from mlops.tasks.base import Task

logger = logging.getLogger(__name__)

REPORT_NAME = "report.json"
# 내보내기 확인에 쓰는 검증 구간 표본 수
EXPORT_CHECK_ROWS = 32
# 실수 예측(회귀)은 float32 변환 오차만 허용한다
FLOAT_RTOL = 1e-4
FLOAT_ATOL = 1e-3
# 라벨 예측(분류)은 거의 모두 같아야 한다. 동점에 가까운 로짓이 연산 순서 차이로 뒤집히는 정도
# (32행 중 1행)만 허용한다
MIN_LABEL_AGREEMENT = 0.95
TRUE_WORDS = frozenset({"true", "1", "yes"})
FALSE_WORDS = frozenset({"false", "0", "no"})


class ExportMismatchError(RuntimeError):
    """내보낸 모델의 예측이 학습 때 모델과 다르다."""


def _cast(key: str, raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        if raw.lower() in TRUE_WORDS:
            return True
        if raw.lower() in FALSE_WORDS:
            return False
        raise ValueError(f"{key}: true 또는 false여야 한다: {raw!r}")
    if isinstance(default, str):
        return raw
    if default is None:
        return _parse_json_or_text(raw)
    try:
        return type(default)(raw)
    except ValueError as error:
        raise ValueError(f"{key}: {type(default).__name__} 값이 아니다: {raw!r}") from error


def _parse_json_or_text(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_params(defaults: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """`key=value` 목록을 기본값의 타입에 맞춰 읽고 기본값과 합친 새 dict를 돌려준다."""
    parsed = {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"파라미터는 key=value 형식이다: {item!r}")
        if key not in defaults:
            raise ValueError(f"알 수 없는 파라미터: {key!r}. 쓸 수 있는 것: {', '.join(defaults)}")
        parsed[key] = _cast(key, raw, defaults[key])
    return {**defaults, **parsed}


def sample_rows(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """n행을 고정 시드로 뽑는다. 시간 순서는 유지한다."""
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


def compare_predictions(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected, actual = np.asarray(expected), np.asarray(actual)
    rows = len(expected)
    if expected.shape != actual.shape:
        return {
            "kind": "shape",
            "expected": list(expected.shape),
            "actual": list(actual.shape),
            "rows": rows,
            "passed": False,
        }
    if np.issubdtype(expected.dtype, np.floating):
        max_diff = float(np.abs(expected - actual.astype(expected.dtype)).max()) if rows else 0.0
        return {
            "kind": "float",
            # NaN은 JSON 표준이 아니라서 None으로 남긴다. allclose가 NaN을 실패로 판정한다
            "max_abs_diff": max_diff if math.isfinite(max_diff) else None,
            "rows": rows,
            "passed": bool(np.allclose(actual, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)),
        }
    agreement = float(np.mean(expected == actual)) if rows else 1.0
    return {
        "kind": "label",
        "agreement": agreement,
        "rows": rows,
        "passed": agreement >= MIN_LABEL_AGREEMENT,
    }


def run_prepare(task: Task, data_dir: Path) -> Path:
    slices = task.prepare()
    if tuple(slices) != task.slices:
        raise ValueError(
            f"{task.name}.prepare()가 돌려준 슬라이스 {tuple(slices)}가 정의 {task.slices}와 다르다"
        )
    for name, df in slices.items():
        logger.info("%s/%s: %d행", task.name, name, len(df))
    return write_slices(data_dir, task.name, slices, task.source)


def _peak_rss_mb() -> float:
    # Linux의 ru_maxrss 단위는 KB. 프로세스가 시작된 뒤의 최대값이다
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"JSON으로 쓸 수 없는 값: {value!r}")


def _evaluate_slices(task: Task, model: Any, data_dir: Path, rows: int | None) -> dict[str, Any]:
    metrics = {}
    for name in task.slices:
        df = sample_rows(read_slice(data_dir, task.name, name), rows)
        metrics[name] = task.evaluate(df, task.predict(model, df))
        logger.info("평가 %s: %s", name, metrics[name])
    return metrics


def resolve_split(
    task: Task, train_slices: Sequence[str] | None, valid_slices: Sequence[str] | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """학습·검증 슬라이스를 정한다. 주지 않으면 과제의 기본값을 쓴다.

    재학습 실습에서 새 슬라이스가 도착하면 학습 구간을 늘리고 검증 구간을 뒤로 옮긴다.
    """
    train = tuple(train_slices or task.train_slices)
    valid = tuple(valid_slices or task.valid_slices)
    unknown = sorted(set(train + valid) - set(task.slices))
    if unknown:
        raise ValueError(
            f"{task.name}에 없는 슬라이스: {unknown}. 있는 것: {', '.join(task.slices)}"
        )
    overlap = sorted(set(train) & set(valid))
    if overlap:
        raise ValueError(f"학습과 검증 슬라이스가 겹친다: {overlap}")
    return train, valid


def _check_replaceable(out_dir: Path) -> None:
    """이전 학습 결과(report.json이 있는 폴더)나 빈 폴더만 바꾼다. 엉뚱한 폴더를 지우지 않도록."""
    if out_dir.is_file():
        raise ValueError(f"{out_dir}는 파일이다. 결과 폴더 경로를 지정한다")
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / REPORT_NAME).exists():
        raise ValueError(f"{out_dir}에 학습 결과가 아닌 파일이 있다. 빈 폴더나 새 경로를 지정한다")


def _promote(staging: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)


def _fit_and_export(
    task: Task,
    data_dir: Path,
    staging: Path,
    split: tuple[pd.DataFrame, pd.DataFrame],
    params: Mapping[str, Any],
    eval_rows: int | None,
) -> dict[str, Any]:
    """학습한 모델은 이 함수 안에서만 잡고 있다가, 내보낸 파일을 다시 올리기 전에 놓아 준다.

    LLM은 모델 두 벌이 동시에 메모리에 올라가면 상한을 넘을 수 있다.
    """
    train_df, valid_df = split
    started = time.perf_counter()
    model = task.train(train_df, valid_df, params)
    seconds = {"train": time.perf_counter() - started}

    started = time.perf_counter()
    metrics = _evaluate_slices(task, model, data_dir, eval_rows)
    seconds["evaluate"] = time.perf_counter() - started

    started = time.perf_counter()
    check_df = sample_rows(valid_df, EXPORT_CHECK_ROWS)
    expected = task.predict(model, check_df)
    task.export(model, staging)
    del model
    gc.collect()
    export_check = compare_predictions(expected, task.predict_exported(staging, check_df))
    seconds["export"] = time.perf_counter() - started
    return {"metrics": metrics, "export_check": export_check, "seconds": seconds}


def run_training(
    task: Task,
    data_dir: Path,
    out_dir: Path,
    params: Mapping[str, Any] | None = None,
    max_eval_rows: int | None = None,
    train_slices: Sequence[str] | None = None,
    valid_slices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """학습하고 모든 슬라이스를 평가한 뒤 내보낸다. 결과는 out_dir/report.json에도 쓴다.

    결과는 임시 폴더에 쓰고 끝까지 돌았을 때만 out_dir로 바꾼다.
    도중에 죽으면 이전 결과가 그대로 남는다.
    max_eval_rows가 None이면 과제의 기본값(task.max_eval_rows)을 쓴다.
    """
    unknown = set(params or {}) - set(task.default_params)
    if unknown:
        raise ValueError(f"알 수 없는 파라미터: {sorted(unknown)}")
    resolved = {**task.default_params, **(params or {})}
    eval_rows = task.max_eval_rows if max_eval_rows is None else max_eval_rows
    train, valid = resolve_split(task, train_slices, valid_slices)
    out_dir = Path(out_dir).resolve()
    _check_replaceable(out_dir)

    split = (read_slices(data_dir, task.name, train), read_slices(data_dir, task.name, valid))
    logger.info(
        "학습 %s: train %s %d행, valid %s %d행, 파라미터 %s",
        task.name,
        train,
        len(split[0]),
        valid,
        len(split[1]),
        resolved,
    )
    staging = out_dir.parent / f".{out_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    result = _fit_and_export(task, data_dir, staging, split, resolved, eval_rows)

    report = {
        "task": task.name,
        "params": resolved,
        "train_slices": list(train),
        "valid_slices": list(valid),
        "max_eval_rows": eval_rows,
        "data": read_manifest(data_dir, task.name),
        "metrics": result["metrics"],
        "export_check": result["export_check"],
        "seconds": {k: round(v, 1) for k, v in result["seconds"].items()},
        "peak_rss_mb": _peak_rss_mb(),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, default=_to_builtin, allow_nan=False)
    (staging / REPORT_NAME).write_text(text + "\n")
    # 내보내기 확인에 실패해도 끝까지 돈 실행이므로 결과를 남긴다.
    # 성공 여부는 report.json의 export_check로 구분한다
    _promote(staging, out_dir)
    if not report["export_check"]["passed"]:
        raise ExportMismatchError(
            f"내보낸 모델의 예측이 학습 때와 다르다: {report['export_check']}"
        )
    return json.loads(text)
