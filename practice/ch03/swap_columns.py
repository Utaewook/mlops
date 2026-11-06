"""3편 부숴 보기: 열 순서를 바꿔 넣으면 모델은 무엇을 하는가.

    bash practice/ch03/run.sh python practice/ch03/swap_columns.py

champion 모델에 2017Q3를 여러 모양으로 넣어 본다.
    1. 학습 때와 같은 열 순서        → 정상 예측
    2. 두 열의 순서를 바꾼 행렬      → 오류 없이 다른 값이 나온다(조용한 오답)
    3. float64 행렬, 열 하나 뺀 행렬 → 거부된다

이 모델의 입력 형식(signature)은 "float32 텐서 [-1, 15]"다. 열 이름 없이 숫자 행렬만 받기 때문에
MLflow가 검사할 수 있는 것은 타입과 열 개수뿐이다. 순서가 바뀌었는지는 알 수 없다.

4. 같은 모델을 열 이름이 있는 형식(DataFrame 입력 예시 → 열마다 이름과 타입이 적힌 signature)으로
임시 폴더에 저장해 다시 불러 본다. 이 형식에서는 MLflow가 열을 이름으로 맞춰 넣으므로 순서를 바꿔도 결과가 같고,
열이 빠지면 거부한다. 임시 폴더에만 저장하므로 실험 기록과 레지스트리에는 아무것도 남지 않는다.
"""

import argparse
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mlops.config import DATA_DIR
from mlops.slices import read_slice
from mlops.tasks.timeseries import CURRENT_OT, FEATURES, TASK, feature_matrix

DEFAULT_MODEL_NAME = "etth1-ot"
DEFAULT_ALIAS = "champion"
DEFAULT_SLICE = "2017Q3"
# 바꿔 넣을 두 열. 고부하 유효 전력(HUFL, 약 -20~23)과 지금 오일 온도(OT_lag0, 약 6~28)는
# 값의 범위가 겹쳐서, 행렬만 봐서는 바뀐 것을 알아채기 어렵다
SWAP = ("HUFL", "OT_lag0")
# MLflow의 거부 메시지는 앞부분에 입력 데이터를 그대로 찍고, 이유는 이 문구 뒤에 적는다
REASON_MARKER = "Error: "
REASON_MAX_CHARS = 110
LABEL_WIDTH = 44


def mae_of(df: pd.DataFrame, change: Any) -> float:
    # 모델은 6시간 동안의 변화량을 낸다. 지금 온도를 더해 6시간 뒤 온도로 바꾼 뒤 정답과 비교한다
    predicted = df[CURRENT_OT].to_numpy() + np.asarray(change, dtype=np.float64).reshape(-1)
    return float(np.abs(df["target"].to_numpy() - predicted).mean())


def try_predict(label: str, predict: Callable[[], Any], df: pd.DataFrame) -> float | None:
    """예측해서 MAE를 보여 주고 돌려준다. MLflow가 입력을 거부하면 이유를 보여 주고 None을 돌려준다."""
    try:
        mae = mae_of(df, predict())
    except mlflow.exceptions.MlflowException as error:
        reason = str(error).rsplit(REASON_MARKER, 1)[-1].strip()
        print(f"{label:<{LABEL_WIDTH}} 거부: {reason[:REASON_MAX_CHARS]}")
        return None
    print(f"{label:<{LABEL_WIDTH}} MAE {mae:.4f}")
    return mae


def swapped_order() -> list[str]:
    """FEATURES에서 SWAP의 두 열만 자리를 바꾼 새 목록."""
    order = list(FEATURES)
    first, second = (order.index(name) for name in SWAP)
    order[first], order[second] = order[second], order[first]
    return order


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="열 순서를 바꿔 넣었을 때 모델이 어떻게 하는지 본다"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="레지스트리의 모델 이름")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="불러올 별칭")
    parser.add_argument("--slice", default=DEFAULT_SLICE, help=f"예측할 슬라이스. {TASK.slices}")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="슬라이스가 있는 폴더")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, float | None]:
    args = parse_args(argv)
    version = MlflowClient().get_model_version_by_alias(args.model_name, args.alias)
    uri = f"models:/{args.model_name}/{version.version}"
    model = mlflow.pyfunc.load_model(uri)
    print(f"{args.model_name}@{args.alias} = 버전 {version.version}")
    print(f"입력 형식: {model.metadata.get_input_schema()}\n")

    df = read_slice(args.data_dir, TASK.name, args.slice)
    in_order = list(FEATURES)
    swapped = swapped_order()
    results: dict[str, float | None] = {}

    print(f"== 1~3. 열 이름이 없는 형식(지금 등록된 모델), {args.slice}")
    results["same_order"] = try_predict(
        "1. 학습 때와 같은 순서", lambda: model.predict(feature_matrix(df)), df
    )
    results["swapped"] = try_predict(
        f"2. {SWAP[0]}와 {SWAP[1]}를 바꾼 순서",
        lambda: model.predict(df[swapped].to_numpy(dtype=np.float32)),
        df,
    )
    results["float64"] = try_predict(
        "3a. float64 행렬", lambda: model.predict(df[in_order].to_numpy(dtype=np.float64)), df
    )
    results["missing_column"] = try_predict(
        f"3b. 열 하나를 뺀 행렬({len(in_order) - 1}개)",
        lambda: model.predict(df[in_order[:-1]].to_numpy(dtype=np.float32)),
        df,
    )

    print("\n== 4. 같은 모델을 열 이름이 있는 형식으로 저장해 보면")
    booster = mlflow.lightgbm.load_model(uri)
    as_frame = df[in_order].astype(np.float32)
    with tempfile.TemporaryDirectory() as folder:
        # DataFrame을 입력 예시로 주면 signature가 "열 이름: 타입" 목록이 된다
        mlflow.lightgbm.save_model(booster, f"{folder}/named", input_example=as_frame.head(5))
        named = mlflow.pyfunc.load_model(f"{folder}/named")
        print(f"입력 형식: {named.metadata.get_input_schema()}\n")
        results["named_same_order"] = try_predict(
            "4a. 같은 순서의 DataFrame", lambda: named.predict(as_frame), df
        )
        results["named_swapped"] = try_predict(
            "4b. 순서를 바꾼 DataFrame", lambda: named.predict(as_frame[swapped]), df
        )
        results["named_missing_column"] = try_predict(
            f"4c. {SWAP[1]} 열이 빠진 DataFrame",
            lambda: named.predict(as_frame.drop(columns=[SWAP[1]])),
            df,
        )
    return results


if __name__ == "__main__":
    main()
