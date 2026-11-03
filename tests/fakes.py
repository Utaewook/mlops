"""테스트용 가짜 과제. 네트워크나 무거운 라이브러리 없이 runner와 CLI를 시험한다."""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MeanTask:
    """runner를 오프라인으로 시험하는 가짜 과제. 학습 구간의 평균을 예측한다."""

    broken_export: bool = False
    name: str = "mean"
    slices: tuple[str, ...] = ("s1", "s2", "s3")
    train_slices: tuple[str, ...] = ("s1",)
    valid_slices: tuple[str, ...] = ("s2",)
    max_eval_rows: int | None = None

    @property
    def default_params(self):
        return {"shift": 0.0}

    @property
    def source(self):
        return {"kind": "synthetic"}

    def prepare(self):
        return {
            "s1": pd.DataFrame({"y": [1.0, 2.0, 3.0]}),
            "s2": pd.DataFrame({"y": [2.0, 3.0]}),
            "s3": pd.DataFrame({"y": [10.0, 11.0]}),
        }

    def train(self, train, valid, params):
        return float(train["y"].mean()) + params["shift"]

    def predict(self, model, df):
        return np.full(len(df), model)

    def evaluate(self, df, predictions):
        return {"mae": float(np.abs(df["y"].to_numpy() - predictions).mean()), "rows": len(df)}

    def export(self, model, out_dir):
        value = model + 1.0 if self.broken_export else model
        (out_dir / "model.json").write_text(json.dumps({"value": value}))

    def predict_exported(self, out_dir, df):
        value = json.loads((out_dir / "model.json").read_text())["value"]
        return np.full(len(df), value)
