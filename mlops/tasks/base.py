"""세 유형의 과제가 공통으로 따르는 인터페이스.

runner와 이후 편의 MLflow 기록, Triton 배포, Dagster 파이프라인, 드리프트 모니터링은
이 인터페이스만 보고 동작한다.
새 과제를 더하려면 이 메서드를 갖춘 객체를 `mlops/tasks/<이름>.py`의 `TASK`로 두고 등록한다.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class Task(Protocol):
    name: str
    # 시간 순서대로 나열한 슬라이스 이름. 학습·검증에 쓰지 않은 뒤쪽 슬라이스는 "미래" 데이터다
    slices: tuple[str, ...]
    train_slices: tuple[str, ...]
    valid_slices: tuple[str, ...]
    # 슬라이스마다 평가할 최대 행 수. 추론이 느린 과제(LLM)는 표본만 평가한다. None이면 전부
    max_eval_rows: int | None

    @property
    def default_params(self) -> Mapping[str, Any]: ...

    @property
    def source(self) -> Mapping[str, Any]:
        """원본 데이터의 고정된 출처(URL·리비전·라이선스). manifest에 남는다."""
        ...

    def prepare(self) -> dict[str, pd.DataFrame]:
        """원본을 받아 슬라이스 이름 → DataFrame을 `slices` 순서대로 돌려준다."""
        ...

    def train(self, train: pd.DataFrame, valid: pd.DataFrame, params: Mapping[str, Any]) -> Any: ...

    def predict(self, model: Any, df: pd.DataFrame) -> np.ndarray:
        """회귀는 실수, 분류는 라벨을 행마다 하나씩 돌려준다."""
        ...

    def evaluate(self, df: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]: ...

    def export(self, model: Any, out_dir: Path) -> None:
        """서빙용 파일(ONNX 또는 어댑터)과 전처리 정보를 out_dir에 쓴다."""
        ...

    def predict_exported(self, out_dir: Path, df: pd.DataFrame) -> np.ndarray:
        """내보낸 파일만으로 예측한다. 학습 때의 예측과 같아야 한다."""
        ...
