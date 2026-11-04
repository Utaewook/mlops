"""Time Series 과제: ETTh1 변압기 데이터로 6시간 뒤 오일 온도(OT)를 예측한다 (LightGBM).

- 데이터: ETDataset의 ETTh1 (2016-07 ~ 2018-06, 1시간 간격). 라이선스 CC BY-ND 4.0이라
  원본과 가공 데이터를 저장소에 올리지 않는다. prepare가 고정된 커밋에서 받아 온다
- 슬라이스: 예측 대상 시각(target_time)의 분기. 2016Q3~2017Q1로 학습, 2017Q2로 검증, 나머지는 미래
- 모델은 온도 자체가 아니라 지금 온도에서 얼마나 변할지를 학습한다. 트리 모델은 학습 때 본 범위 밖의
  값을 내지 못해서, 겨울처럼 온도 수준이 바뀌면 절대값 예측이 크게 틀린다
- 기준선은 "6시간 뒤에도 지금 온도 그대로"(persistence)다. 24시간 뒤는 하루 주기와 겹쳐서
  어떤 설정도 이 기준선을 넘지 못했다(검증 MAE 2.42 대 2.46).
  6시간 뒤는 하루 안의 변화를 배울 수 있다(1.79 대 2.23)
- 조기 종료에 검증 슬라이스를 쓰므로 검증 지표는 조금 낙관적이다.
  정직한 성능은 미래 슬라이스에서 본다
- 내보내기: ONNX(model.onnx)와 입력 특성 순서(features.json). ONNX 그래프 안에서 지금 온도를 더해
  예측 온도를 바로 내놓는다
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from mlops.download import fetch_url

logger = logging.getLogger(__name__)

ETT_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    "1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv"
)
ETT_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"

LOAD_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL")
HORIZON_HOURS = 6
OT_LAGS = (0, 1, 2, 3, 6, 12, 24)
FEATURES = (*LOAD_COLUMNS, *(f"OT_lag{lag}" for lag in OT_LAGS), "hour", "dayofweek")
SLICES = ("2016Q3", "2016Q4", "2017Q1", "2017Q2", "2017Q3", "2017Q4", "2018Q1", "2018Q2")
# onnxmltools 1.16의 LightGBM 변환기가 지원하는 최대 opset
ONNX_OPSET = 15
ONNX_INPUT = "features"
ONNX_OUTPUT = "forecast"
CURRENT_OT = "OT_lag0"


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """시각 t의 값으로 특성을 만들고 t+HORIZON_HOURS의 OT를 target으로 붙인다.

    특성은 t 이전 값만 쓴다. 이력이나 target이 없는 처음과 끝 행은 버린다.
    """
    df = raw.sort_values("date").reset_index(drop=True)
    if not (df["date"].diff().dropna() == pd.Timedelta(1, unit="h")).all():
        raise ValueError("시간 간격이 1시간이 아닌 행이 있다. 빠진 시각을 먼저 확인한다")

    columns = {
        "timestamp": df["date"],
        "target_time": df["date"] + pd.Timedelta(HORIZON_HOURS, unit="h"),
        **{column: df[column] for column in LOAD_COLUMNS},
        **{f"OT_lag{lag}": df["OT"].shift(lag) for lag in OT_LAGS},
        "hour": df["date"].dt.hour,
        "dayofweek": df["date"].dt.dayofweek,
        "target": df["OT"].shift(-HORIZON_HOURS),
    }
    return pd.DataFrame(columns).dropna().reset_index(drop=True)


def split_by_quarter(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """target_time의 분기로 나눈다.

    정답이 그 분기 안에서 나오므로 앞 슬라이스의 정답이 뒤 슬라이스 기간으로 넘어가지 않는다.
    """
    target_time = features["target_time"]
    quarters = target_time.dt.year.astype(str) + "Q" + target_time.dt.quarter.astype(str)
    unknown = sorted(set(quarters) - set(SLICES))
    if unknown:
        raise ValueError(f"정의되지 않은 분기의 데이터가 있다: {unknown}")
    return {name: features[quarters == name].reset_index(drop=True) for name in SLICES}


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """모델 입력 행렬. 열 순서는 FEATURES다.

    ONNX 모델의 입력이 float32라서 학습과 예측도 같은 값으로 한다.
    """
    return df[list(FEATURES)].to_numpy(dtype=np.float32)


def _change(df: pd.DataFrame) -> np.ndarray:
    return (df["target"] - df[CURRENT_OT]).to_numpy()


def _add_current_temperature(onnx_model: Any) -> Any:
    """LightGBM 출력(변화량)에 입력의 지금 온도 열을 더하는 노드를 그래프 끝에 붙인다."""
    from onnx import TensorProto, checker, helper, numpy_helper

    # 변환기는 트리 연산(ai.onnx.ml)만 써서 기본 도메인 opset이 낮다. Slice 입력 형식에 맞게 올린다
    default_domain = [o for o in onnx_model.opset_import if o.domain in ("", "ai.onnx")]
    for opset in default_domain:
        opset.version = max(opset.version, ONNX_OPSET)
    if not default_domain:
        onnx_model.opset_import.append(helper.make_opsetid("", ONNX_OPSET))

    graph = onnx_model.graph
    change = graph.output[0].name
    column = FEATURES.index(CURRENT_OT)
    graph.initializer.extend(
        numpy_helper.from_array(np.array([value], dtype=np.int64), name)
        for name, value in [("ot_start", column), ("ot_end", column + 1), ("ot_axis", 1)]
    )
    graph.node.extend(
        [
            helper.make_node("Slice", [ONNX_INPUT, "ot_start", "ot_end", "ot_axis"], ["ot_now"]),
            helper.make_node("Add", [change, "ot_now"], [ONNX_OUTPUT]),
        ]
    )
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(ONNX_OUTPUT, TensorProto.FLOAT, [None, 1]))
    checker.check_model(onnx_model)
    return onnx_model


@dataclass(frozen=True)
class TimeSeriesTask:
    name: str = "timeseries"
    slices: tuple[str, ...] = SLICES
    train_slices: tuple[str, ...] = ("2016Q3", "2016Q4", "2017Q1")
    valid_slices: tuple[str, ...] = ("2017Q2",)
    max_eval_rows: int | None = None

    @property
    def default_params(self) -> Mapping[str, Any]:
        return {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "early_stopping_rounds": 100,
            "seed": 42,
        }

    @property
    def source(self) -> Mapping[str, Any]:
        return {"url": ETT_URL, "sha256": ETT_SHA256, "license": "CC BY-ND 4.0"}

    def prepare(self) -> dict[str, pd.DataFrame]:
        raw = pd.read_csv(fetch_url(ETT_URL, ETT_SHA256), parse_dates=["date"])
        return split_by_quarter(build_features(raw))

    def train(
        self, train: pd.DataFrame, valid: pd.DataFrame, params: Mapping[str, Any]
    ) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=params["min_child_samples"],
            random_state=params["seed"],
            # 조기 종료는 MAE(l1) 하나로만 판단한다. 같은 데이터·시드면 같은 모델이 나오게 한다
            metric="l1",
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        model.fit(
            feature_matrix(train),
            _change(train),
            eval_X=(feature_matrix(valid),),
            eval_y=(_change(valid),),
            callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
        )
        logger.info("best iteration: %s", model.best_iteration_)
        return model

    def predict(self, model: lgb.LGBMRegressor, df: pd.DataFrame) -> np.ndarray:
        return df[CURRENT_OT].to_numpy() + model.predict(feature_matrix(df))

    def evaluate(self, df: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
        """skill = 1 - mae / naive_mae. 0보다 크면 지금 온도를 그대로 쓰는 것보다 낫다."""
        error = df["target"].to_numpy() - predictions
        mae = float(np.abs(error).mean())
        naive_mae = float(np.abs(_change(df)).mean())
        return {
            "mae": mae,
            "rmse": float(np.sqrt((error**2).mean())),
            "naive_mae": naive_mae,
            "skill": 1.0 - mae / naive_mae if naive_mae > 0 else 0.0,
            "rows": len(df),
        }

    def export(self, model: lgb.LGBMRegressor, out_dir: Path) -> None:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = onnxmltools.convert_lightgbm(
            model,
            initial_types=[(ONNX_INPUT, FloatTensorType([None, len(FEATURES)]))],
            target_opset=ONNX_OPSET,
        )
        (out_dir / "model.onnx").write_bytes(
            _add_current_temperature(onnx_model).SerializeToString()
        )
        spec = {
            "input": ONNX_INPUT,
            "output": ONNX_OUTPUT,
            "features": list(FEATURES),
            "horizon_hours": HORIZON_HOURS,
        }
        (out_dir / "features.json").write_text(json.dumps(spec, indent=2) + "\n")

    def predict_exported(self, out_dir: Path, df: pd.DataFrame) -> np.ndarray:
        import onnxruntime as ort

        spec = json.loads((out_dir / "features.json").read_text())
        session = ort.InferenceSession(
            str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        features = df[spec["features"]].to_numpy(dtype=np.float32)
        (output,) = session.run([spec["output"]], {spec["input"]: features})
        return output.ravel().astype(np.float64)


TASK = TimeSeriesTask()
