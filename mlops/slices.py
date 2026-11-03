"""시간 순서가 있는 데이터 조각(슬라이스) 저장소.

과제마다 `<data_dir>/<task>/<slice>.parquet`와 `manifest.json`을 둔다.
슬라이스는 "데이터가 시간에 따라 도착한다"는 상황을 흉내 낸다. 앞쪽 슬라이스로 학습하고,
뒤쪽 슬라이스로 성능 저하(드리프트)를 보거나 재학습 데이터로 쓴다.
"""

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_NAME = "manifest.json"
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _checked(name: str) -> str:
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"슬라이스 이름에는 영문, 숫자, -, _만 쓴다: {name!r}")
    return name


def slice_path(data_dir: Path, task: str, name: str) -> Path:
    return Path(data_dir) / _checked(task) / f"{_checked(name)}.parquet"


def write_slices(
    data_dir: Path, task: str, slices: Mapping[str, pd.DataFrame], source: Mapping[str, Any]
) -> Path:
    """슬라이스를 순서대로 저장하고 manifest 경로를 돌려준다."""
    paths = {name: slice_path(data_dir, task, name) for name in slices}
    task_dir = Path(data_dir) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    for name, df in slices.items():
        df.to_parquet(paths[name], index=False)

    manifest = {
        "task": task,
        "source": dict(source),
        "slices": [{"name": name, "rows": len(df)} for name, df in slices.items()],
    }
    manifest_path = task_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def read_slice(data_dir: Path, task: str, name: str) -> pd.DataFrame:
    path = slice_path(data_dir, task, name)
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없다. 먼저 `python -m mlops prepare {task}`를 돌린다")
    return pd.read_parquet(path)


def read_slices(data_dir: Path, task: str, names: Iterable[str]) -> pd.DataFrame:
    """여러 슬라이스를 순서대로 이어 붙인다."""
    return pd.concat([read_slice(data_dir, task, n) for n in names], ignore_index=True)


def read_manifest(data_dir: Path, task: str) -> dict[str, Any]:
    path = Path(data_dir) / _checked(task) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없다. 먼저 `python -m mlops prepare {task}`를 돌린다")
    return json.loads(path.read_text())
