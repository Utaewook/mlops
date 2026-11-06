"""3편 실습 1: 학습 한 번을 MLflow run 하나로 기록한다.

    bash practice/ch03/run.sh python practice/ch03/train.py                       # 직접 기록
    bash practice/ch03/run.sh python practice/ch03/train.py --param learning_rate=0.1
    bash practice/ch03/run.sh python practice/ch03/train.py --autolog             # autolog로 기록

MLflow가 기록을 담는 단위는 다음과 같다.

    실험(experiment) "etth1-ot-6h"
     ├─ run  ← start_run() 한 번이 run 하나. 학습 한 번의 기록 묶음
     │   ├─ 파라미터(params)  학습 전에 정한 값. 문자열로 저장되고, 한 번 쓰면 바꿀 수 없다
     │   ├─ 지표(metrics)     학습 결과 숫자. 값마다 step(몇 번째)과 시각이 붙어 이력이 남는다
     │   ├─ 태그(tags)        자유 메모. MLflow가 스스로 붙이는 mlflow.* 태그(git 커밋 등)도 여기에 있다
     │   ├─ 입력(inputs)      어떤 데이터로 학습했는가. 데이터 자체가 아니라 이름, 지문(digest), 열 구조만
     │   └─ 산출물(artifacts) 파일
     └─ 모델(LoggedModel)  ← MLflow 3.x부터 run과 나란한 별도 객체다. id가 m-로 시작한다.
                              어느 run에서 나왔는지(source_run_id)를 기억하고, 레지스트리에는 이것을 등록한다

run.sh가 MLFLOW_TRACKING_URI를 비워 두므로 기록은 저장소 루트의 두 곳에 쌓인다.
    mlflow.db  SQLite 파일. 실험, run, 파라미터, 지표, 태그, 모델 정보, 레지스트리가 모두 표로 들어간다
    mlruns/    파일. 모델 파일은 mlruns/<실험 id>/models/<모델 id>/artifacts/ 아래에 있다
무엇이 어디에 들어갔는지는 inspect_store.py로 볼 수 있다.
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Metric

from mlops.config import DATA_DIR
from mlops.runner import parse_params
from mlops.slices import read_slices
from mlops.tasks.timeseries import TASK, feature_matrix

EXPERIMENT_NAME = "etth1-ot-6h"
# run 안에서 모델을 부르는 이름. MLflow 2.x의 artifact_path 자리를 3.x에서는 name이 대신한다
MODEL_NAME_IN_RUN = "model"
# 모델과 같이 저장할 입력 예시 행 수. MLflow는 이것으로 입력 형식(signature)도 추론한다
INPUT_EXAMPLE_ROWS = 5
# LightGBM이 조기 종료용 검증 세트에 붙이는 이름과 지표 이름(metric="l1"은 MAE)
EVAL_SET_NAME = "valid_0"
EVAL_METRIC = "l1"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETTh1 6시간 뒤 오일 온도 예측 모델을 학습하고 기록한다"
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"바꿀 하이퍼파라미터. 쓸 수 있는 것: {', '.join(TASK.default_params)}",
    )
    parser.add_argument("--autolog", action="store_true", help="직접 기록 대신 autolog로 기록한다")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="기록할 실험 이름")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="슬라이스가 있는 폴더")
    return parser.parse_args(argv)


def run_name(params: Mapping[str, Any], autolog: bool) -> str:
    # run 이름은 UI 목록에서 알아보기 위한 것일 뿐이다. 겹쳐도 되고, run은 run_id로 구분한다
    mode = "autolog" if autolog else "manual"
    return f"{mode}-lr{params['learning_rate']}-leaves{params['num_leaves']}"


def log_params_and_inputs(
    params: Mapping[str, Any], train: pd.DataFrame, valid: pd.DataFrame
) -> None:
    """학습 전에 정해지는 것(하이퍼파라미터, 데이터 구간, 데이터 지문)을 남긴다.

    학습보다 먼저 기록해 두면, 학습이 도중에 죽어도 무엇을 시도하다 실패했는지가 run에 남는다.
    """
    # params 표에 (run_uuid, key, value) 행으로 들어간다. 값은 모두 문자열로 바뀐다(0.05 → "0.05").
    # 그래서 검색할 때도 파라미터는 크기 비교를 못 하고 문자열로만 비교한다(params.learning_rate = '0.05')
    mlflow.log_params(params)
    mlflow.log_params(
        {"train_slices": ",".join(TASK.train_slices), "valid_slices": ",".join(TASK.valid_slices)}
    )

    # 데이터 자체는 올리지 않고 요약만 남긴다: 이름, 지문(내용 해시), 열 이름과 타입, 행 수.
    # 같은 이름인데 지문이 다르면 "데이터가 바뀌었다"는 뜻이라 재현성을 따질 때 쓴다.
    # datasets 표에 요약이, inputs 표에 run과의 연결이, input_tags 표에 용도(context)가 들어가고
    # UI의 run 목록 Dataset 칸과 run 화면 Datasets 항목에 보인다.
    # 열 구조를 추론하면서 "정수 열은 결측값을 못 담는다"는 UserWarning을 내는데, 데이터 요약에는 상관없다
    for df, slices, context in (
        (train, TASK.train_slices, "training"),
        (valid, TASK.valid_slices, "validation"),
    ):
        dataset = mlflow.data.from_pandas(df, name=f"etth1-{'+'.join(slices)}", targets="target")
        mlflow.log_input(dataset, context=context)


def log_learning_curve(run_id: str, model: lgb.LGBMRegressor) -> None:
    """LightGBM이 나무를 하나 더할 때마다 잰 검증 MAE를 step별 지표로 남긴다.

    지표는 같은 이름으로 여러 번 기록할 수 있고, step이 다르면 이력으로 쌓인다.
    UI의 run 화면 Model metrics 탭에서 step을 가로축으로 한 곡선으로 보인다.
    (metrics 표에는 모든 step이, latest_metrics 표에는 마지막 값만 들어간다)
    autolog는 이 곡선을 valid_0-l1이라는 이름으로 알아서 남긴다.

    조기 종료 때문에 곡선은 가장 좋았던 나무(best_iteration)에서 early_stopping_rounds(기본 100)개를
    더 간 곳에서 끝난다. 그래서 이 지표의 마지막 값은 mae보다 조금 크다. autolog는 best_iteration이
    있으면(검증 세트를 주면 조기 종료가 없어도 정해진다) 그 시점의 값을 곡선 끝에 step 하나를 더해
    다시 기록하므로, autolog run에서는 마지막 값이 mae와 같다.
    (변화량의 MAE가 곧 온도의 MAE다. 예측과 정답에 같은 '지금 온도'를 더하기 때문이다)
    """
    history = model.evals_result_[EVAL_SET_NAME][EVAL_METRIC]
    now_ms = int(time.time() * 1000)
    # log_metric을 수백 번 부르면 그만큼 DB에 따로 쓴다. log_batch는 한 번에 쓴다
    curve = [Metric("valid_l1_by_tree", value, now_ms, step) for step, value in enumerate(history)]
    MlflowClient().log_batch(run_id, metrics=curve)


def log_results(
    run_id: str, model: lgb.LGBMRegressor, metrics: Mapping[str, float], valid: pd.DataFrame
) -> Any:
    """학습 뒤에 나오는 것(지표, 학습 곡선, 모델)을 남기고 모델 정보를 돌려준다."""
    # 지표 하나가 (key, value, step=0, timestamp) 행 하나가 된다
    mlflow.log_metrics({**metrics, "best_iteration": model.best_iteration_})
    log_learning_curve(run_id, model)

    # 모델을 LoggedModel로 남긴다. mlflow.lightgbm은 LightGBM 전용 "flavor"(저장·불러오기 방식)다.
    # 이 한 줄이 하는 일:
    #   1. logged_models 표에 행을 만들고 id(m-...)를 받는다. source_run_id에 지금 run이 들어간다
    #   2. mlruns/<실험 id>/models/<모델 id>/artifacts/에 파일을 쓴다
    #        model.skops          학습된 LGBMRegressor. pickle 대신 skops 형식이라, 불러올 때
    #                             MLmodel에 적힌 허용 타입만 되살린다(pickle은 불러오는 순간 임의 코드가 돌 수 있다)
    #        MLmodel              이 폴더를 어떻게 불러오는지 적은 설명서. flavor(lightgbm, python_function),
    #                             입력·출력 형식(signature), 만든 run_id, MLflow·Python 버전
    #        requirements.txt, python_env.yaml, conda.yaml
    #                             불러올 때 필요한 패키지와 버전. 만드는 방법이 둘이다.
    #                             - 지금 폴더에 uv.lock과 pyproject.toml이 있으면(저장소 루트에서 돌리면 그렇다)
    #                               `uv export`로 프로젝트 전체 의존성을 적는다. 이 저장소에서는 torch, dagster까지
    #                               205줄이다. LightGBM 모델 하나에 필요한 것보다 훨씬 많다
    #                             - 아니면(또는 MLFLOW_UV_AUTO_DETECT=false) 모델을 한 번 불러 보며 import되는
    #                               패키지를 추론한다. 12줄이지만 여기에도 필요 없는 torch, pytest가 섞인다
    #                             정확히 하려면 log_model에 pip_requirements를 직접 넘긴다
    #        input_example.json, serving_input_example.json
    #                             아래 입력 예시. 뒤의 것은 REST로 서빙할 때 보낼 요청 본문 모양이다
    #   3. 지금 run의 파라미터와 지표를 모델에도 연결한다(logged_model_params, logged_model_metrics 표).
    #      그래서 실험의 Models 화면에서 run을 거치지 않고 바로 모델끼리 지표를 비교할 수 있다
    # 모델 파일은 run 폴더(mlruns/<실험 id>/<run id>/artifacts/)가 아니라 models/ 아래에 있다.
    # run 폴더는 만들어지지도 않는다. UI의 run 화면 Artifacts 탭에 보이는 파일은 run에 연결된 모델의 파일이고,
    # 탭 위에 "You're viewing artifacts assigned to a logged model associated with this run."이라고 나온다
    #
    # input_example을 주면 MLflow가 입력 형식(signature)을 추론한다. 여기서는 float32 행렬이라
    # MLmodel에 "입력: float32 텐서 [-1, 15], 출력: float64 텐서 [-1]"로 적힌다.
    # pyfunc로 불러온 모델은 이 형식과 맞지 않는 입력을 거부한다.
    # 이 모델은 온도가 아니라 6시간 동안의 변화량을 예측한다는 점에 주의한다(predict.py 참고)
    return mlflow.lightgbm.log_model(
        model,
        name=MODEL_NAME_IN_RUN,
        input_example=feature_matrix(valid)[:INPUT_EXAMPLE_ROWS],
    )


def train_and_evaluate(
    train: pd.DataFrame, valid: pd.DataFrame, params: Mapping[str, Any]
) -> tuple[lgb.LGBMRegressor, dict[str, float]]:
    # MLflow와 상관없는 부분. 학습은 2016Q3~2017Q1, 조기 종료와 평가는 2017Q2로 한다
    model = TASK.train(train, valid, params)
    metrics = TASK.evaluate(valid, TASK.predict(model, valid))
    return model, metrics


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    params = parse_params(TASK.default_params, args.param)
    train = read_slices(args.data_dir, TASK.name, TASK.train_slices)
    valid = read_slices(args.data_dir, TASK.name, TASK.valid_slices)

    # 실험이 없으면 만든다. experiments 표에 행이 생기고, 산출물 위치로 mlruns/<실험 id>가 정해진다.
    # 이후 start_run()은 이 실험 아래에 run을 만든다
    mlflow.set_experiment(args.experiment)

    if args.autolog:
        # autolog는 lightgbm.train(과 cv, Dataset 생성)을 감싸서(patch) 학습이 불릴 때 파라미터, 지표,
        # 모델, 데이터 요약을 알아서 기록한다. LGBMRegressor.fit도 안에서 lightgbm.train을 부르므로
        # 같이 잡힌다. 그래서 학습보다 먼저 켜야 한다.
        # 대신 fit 밖에서 계산하는 값(아래 TASK.evaluate의 mae, skill 등)은 모른다
        mlflow.lightgbm.autolog()

    try:
        # with 블록이 run 하나다. 들어갈 때 runs 표에 RUNNING 상태로 행이 생기고, 나올 때 FINISHED가 된다.
        # 블록 안에서 예외가 나면 FAILED로 끝난다.
        # run이 시작될 때 MLflow가 스스로 붙이는 태그: mlflow.user(실행한 사용자),
        # mlflow.source.name(실행한 파일), mlflow.source.git.commit(지금 커밋. git이 있어야 한다)
        with mlflow.start_run(run_name=run_name(params, args.autolog)) as run:
            # 태그는 나중에 고칠 수 있는 메모다. 검색 조건으로도 쓴다(register.py가 manual run만 고른다)
            mlflow.set_tags({"task": TASK.name, "logging": "autolog" if args.autolog else "manual"})
            if not args.autolog:
                log_params_and_inputs(params, train, valid)

            model, metrics = train_and_evaluate(train, valid, params)

            model_id = None
            if not args.autolog:
                model_id = log_results(run.info.run_id, model, metrics, valid).model_id
    finally:
        if args.autolog:
            # autolog는 프로세스 전체에 걸린다. 같은 프로세스에서 다른 학습을 할 때 섞이지 않게 끈다
            mlflow.lightgbm.autolog(disable=True)

    summary = {
        "experiment_id": run.info.experiment_id,
        "run_id": run.info.run_id,
        "model_id": model_id,
        "params": params,
        "best_iteration": model.best_iteration_,
        "valid": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
