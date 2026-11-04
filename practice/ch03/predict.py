"""3편 실습 2: 레지스트리에서 별칭(champion)이 가리키는 모델을 불러와 '미래' 슬라이스를 예측한다.

    bash practice/ch03/run.sh python practice/ch03/predict.py
    bash practice/ch03/run.sh python practice/ch03/predict.py --alias challenger --slice 2017Q4

이 코드에는 모델 버전 번호도, run id도 없다. 이름과 별칭만 안다.
register.py로 별칭을 다른 버전으로 옮기고 다시 돌리면, 코드는 그대로인데 다른 모델이 불려 와서
예측과 MAE가 바뀐다. 배포하는 쪽(서빙 코드)과 모델을 고르는 쪽(레지스트리)이 이렇게 나뉜다.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mlops.config import DATA_DIR
from mlops.slices import read_slice
from mlops.tasks.timeseries import CURRENT_OT, HORIZON_HOURS, TASK, feature_matrix

# 학습(2016Q3~2017Q1)과 검증(2017Q2)에 쓰지 않은 첫 분기
DEFAULT_SLICE = "2017Q3"
DEFAULT_MODEL_NAME = "etth1-ot"
DEFAULT_ALIAS = "champion"
DEFAULT_SHOW_ROWS = 5


def load_by_alias(model_name: str, alias: str) -> Any:
    """별칭이 가리키는 버전을 찾아서 그 버전의 모델을 불러온다."""
    # 1. 별칭 → 버전. registered_model_aliases 표에서 (모델 이름, 별칭) → 버전 번호를 찾는다.
    #    돌려받은 ModelVersion에는 그 버전이 어느 run에서 나왔는지(run_id), 실제 모델이
    #    어디 있는지(source = models:/m-...)가 들어 있다
    version = MlflowClient().get_model_version_by_alias(model_name, alias)
    print(
        f"models:/{model_name}@{alias} → 버전 {version.version}"
        f" (run {version.run_id}, 모델 {version.source})"
    )

    # 2. 버전 → 모델. models:/<이름>@<별칭>으로 바로 불러와도 되지만, 그러면 위에서 찾은 뒤에
    #    누가 별칭을 옮겼을 때 출력한 버전과 불러온 모델이 달라질 수 있다. 찾은 버전 번호로 불러온다.
    #
    #    pyfunc는 flavor와 상관없는 공통 인터페이스다. LightGBM이든 PyTorch든 predict(입력) 하나로 쓴다.
    #    MLflow의 서빙 도구(mlflow models serve)도 이 인터페이스로 모델을 돌린다.
    #    불러올 때 MLmodel 파일을 읽어 flavor를 확인하고, 저장할 때 기록한 입력 형식(signature)으로
    #    입력을 검사한다. 원래 객체(LGBMRegressor)가 필요하면 mlflow.lightgbm.load_model을 쓴다
    return mlflow.pyfunc.load_model(f"models:/{model_name}/{version.version}")


def forecast(model: Any, df: pd.DataFrame) -> np.ndarray:
    """모델은 6시간 동안의 변화량을 예측하므로, 지금 온도를 더해야 6시간 뒤 온도가 된다.

    레지스트리는 모델 파일만 관리하고 이런 앞뒤 처리는 모른다. 학습 때와 같은 입력(FEATURES 순서의
    float32 행렬)을 만들고 같은 후처리를 하는 일은 모델을 쓰는 쪽 코드의 몫이다.
    5편에서 ONNX로 내보낼 때는 이 덧셈을 모델 그래프 안에 넣어서 이 부담을 없앤다.
    """
    change = np.asarray(model.predict(feature_matrix(df)), dtype=np.float64).reshape(-1)
    return df[CURRENT_OT].to_numpy() + change


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="별칭이 가리키는 모델로 미래 슬라이스를 예측한다")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="레지스트리의 모델 이름")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="불러올 별칭")
    parser.add_argument("--slice", default=DEFAULT_SLICE, help=f"예측할 슬라이스. {TASK.slices}")
    parser.add_argument("--rows", type=int, default=DEFAULT_SHOW_ROWS, help="화면에 보여 줄 행 수")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="슬라이스가 있는 폴더")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> float:
    args = parse_args(argv)
    model = load_by_alias(args.model_name, args.alias)

    df = read_slice(args.data_dir, TASK.name, args.slice)
    predicted = forecast(model, df)
    table = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "OT_now": df[CURRENT_OT],
            f"forecast_{HORIZON_HOURS}h": predicted.round(3),
            "actual": df["target"].round(3),
        }
    )
    mae = float(np.abs(df["target"].to_numpy() - predicted).mean())
    print(table.head(args.rows).to_string(index=False))
    print(f"\n{args.slice} 전체 {len(df)}행 MAE: {mae:.4f}")
    return mae


if __name__ == "__main__":
    main()
