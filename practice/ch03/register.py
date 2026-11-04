"""3편 실습 2: 모델 레지스트리에 버전을 등록하고, 별칭을 붙이고, 버전에서 run까지 거슬러 올라간다.

    bash practice/ch03/run.sh python practice/ch03/register.py register              # 가장 좋은 run을 등록
    bash practice/ch03/run.sh python practice/ch03/register.py register --run-id <id> # 지정한 run을 등록
    bash practice/ch03/run.sh python practice/ch03/register.py alias 1                 # champion → 버전 1
    bash practice/ch03/run.sh python practice/ch03/register.py list                    # 버전 목록과 출처

레지스트리는 "어떤 모델을 쓸 것인가"를 관리하는 이름표 보관함이다. 모델 파일을 새로 저장하지 않는다.

    등록 모델(registered model) "etth1-ot"   ← registered_models 표
     ├─ 버전 1 → models:/m-aaa (run A에서 나온 모델) ← model_versions 표. 등록할 때마다 번호가 1씩 는다
     ├─ 버전 2 → models:/m-bbb (run B에서 나온 모델)
     └─ 별칭 champion → 버전 2                      ← registered_model_aliases 표. 언제든 옮길 수 있다

등록(register)과 별칭 지정(alias)은 별개의 동작이다. 새 버전을 등록해도 champion은 그대로이고,
사람(또는 검증을 통과한 파이프라인)이 별칭을 옮겨야 모델을 쓰는 쪽이 새 모델을 받는다.
예전의 stage(Staging, Production)는 MLflow 2.9부터 폐기 예정이 되었고, 별칭과 태그가 그 역할을 맡는다.
"""

import argparse
import re
from collections.abc import Sequence

import mlflow
from mlflow import MlflowClient

# 같은 폴더의 train.py. python practice/ch03/register.py로 실행하면 이 폴더가 import 경로에 들어간다
from train import EXPERIMENT_NAME

DEFAULT_MODEL_NAME = "etth1-ot"
DEFAULT_ALIAS = "champion"
# 검색 조건 문자열에 그대로 들어가므로 따옴표 같은 글자는 받지 않는다
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
GIT_COMMIT_TAG = "mlflow.source.git.commit"


def checked(name: str) -> str:
    if not SAFE_NAME.fullmatch(name):
        raise SystemExit(f"이름에는 영문, 숫자, _ . -만 쓴다: {name!r}")
    return name


def best_run_id(experiment: str) -> str:
    """검증 MAE가 가장 낮은 run을 찾는다.

    search_runs의 검색 조건은 SQL WHERE와 비슷하다. 지표는 metrics.<이름>, 태그는 tags.<이름>,
    run 자체의 속성은 attributes.<이름>으로 부른다. 파라미터는 문자열이라 크기 비교를 못 한다.
    autolog run에는 mae 지표가 없으므로 직접 기록한 run(tags.logging = 'manual')만 본다.
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="tags.logging = 'manual' and attributes.status = 'FINISHED'",
        order_by=["metrics.mae ASC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"실험 {experiment!r}에 끝난 manual run이 없다. train.py를 먼저 돌린다")
    run = runs[0]
    print(
        f"가장 좋은 run: {run.info.run_name} ({run.info.run_id}), mae={run.data.metrics['mae']:.4f}"
    )
    return run.info.run_id


def model_uri_of(run_id: str) -> str:
    """run이 남긴 모델(LoggedModel)의 주소를 찾는다.

    MLflow 3.x에서 모델은 run의 산출물이 아니라 별도 객체라서, "이 run에서 나온 모델"을 검색해서 찾는다.
    찾은 주소는 models:/m-... 꼴이다. runs:/<run_id>/model도 받아 주지만, 그 경로에 파일이 없어서
    MLflow가 같은 방식으로 모델을 다시 찾으며 경고를 낸다.
    """
    experiment_id = MlflowClient().get_run(run_id).info.experiment_id
    models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{checked(run_id)}'",
        # run 하나가 모델을 여러 번 남겼을 수도 있으므로 가장 최근 것을 고른다
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        output_format="list",
    )
    if not models:
        raise SystemExit(f"run {run_id}에 기록된 모델이 없다")
    return models[0].model_uri


def register(model_name: str, experiment: str, run_id: str | None) -> None:
    run_id = run_id or best_run_id(experiment)
    model_uri = model_uri_of(run_id)

    # 등록 모델이 없으면 만들고(registered_models), 버전을 하나 추가한다(model_versions).
    # 버전 행에는 모델 주소(source)와 run_id가 같이 저장되어, 버전에서 run으로 거슬러 올라갈 수 있다.
    # 모델 파일은 복사하지 않는다. 버전은 이미 있는 LoggedModel을 가리키는 포인터다
    version = mlflow.register_model(model_uri, model_name)
    print(f"등록: {model_name} 버전 {version.version} ← {model_uri} (run {run_id})")
    print(f"별칭은 바뀌지 않았다. 쓰려면: register.py alias {version.version}")


def set_alias(model_name: str, alias: str, version: str) -> None:
    client = MlflowClient()
    # 지금 별칭이 어느 버전에 붙어 있는지 본다. 등록 모델의 aliases는 {별칭: 버전} 사전이다.
    # (get_model_version_by_alias는 별칭이 없으면 INVALID_PARAMETER_VALUE 예외를 내서, 잘못된 입력과 구분하기 어렵다)
    current = client.get_registered_model(model_name).aliases.get(alias)
    before = "(없음)" if current is None else f"버전 {current}"

    # 별칭은 등록 모델 안에서 하나뿐이다. 다른 버전에 붙어 있었으면 떼어서 이 버전으로 옮긴다.
    # 모델을 쓰는 쪽(predict.py)은 models:/<이름>@<별칭>만 알고 있으므로, 이 한 줄이 곧 "배포 전환"이다.
    # 되돌릴 때도 별칭을 이전 버전으로 다시 옮기면 된다
    client.set_registered_model_alias(model_name, alias, version)
    print(f"{model_name}@{alias}: {before} → 버전 {version}")


def list_versions(model_name: str) -> None:
    """버전마다 어느 run에서 나왔고, 그 run의 지표와 git 커밋이 무엇인지 보여 준다."""
    client = MlflowClient()
    aliases = client.get_registered_model(model_name).aliases  # {별칭: 버전}
    by_version: dict[str, list[str]] = {}
    for alias, version in aliases.items():
        by_version.setdefault(str(version), []).append(alias)

    versions = client.search_model_versions(f"name = '{model_name}'")
    for version in sorted(versions, key=lambda v: int(v.version)):
        # 버전 → run → 파라미터, 지표, 태그(git 커밋). 레지스트리만 보고도 이 모델이 어떤 코드와
        # 설정으로 만들어졌는지 따라갈 수 있다
        run = client.get_run(version.run_id)
        mae = run.data.metrics.get("mae")
        print(
            f"버전 {version.version:>3} {','.join(by_version.get(str(version.version), [])):<12}"
            f" run={run.info.run_name}"
            f" mae={'-' if mae is None else f'{mae:.4f}'}"
            f" lr={run.data.params.get('learning_rate', '-')}"
            f" git={run.data.tags.get(GIT_COMMIT_TAG, '-')[:8]}"
            f" source={version.source}"
        )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="모델 레지스트리 등록, 별칭, 버전 목록")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="등록 모델 이름")
    commands = parser.add_subparsers(dest="command", required=True)

    register_cmd = commands.add_parser("register", help="run의 모델을 새 버전으로 등록한다")
    register_cmd.add_argument("--run-id", help="등록할 run. 없으면 검증 MAE가 가장 낮은 manual run")
    register_cmd.add_argument("--experiment", default=EXPERIMENT_NAME, help="run을 찾을 실험")

    alias_cmd = commands.add_parser("alias", help="별칭을 버전에 붙이거나 옮긴다")
    alias_cmd.add_argument("version", help="별칭을 붙일 버전 번호")
    alias_cmd.add_argument("--alias", default=DEFAULT_ALIAS, help="별칭 이름")

    commands.add_parser("list", help="버전 목록과 각 버전의 출처 run을 보여 준다")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_name = checked(args.model_name)
    if args.command == "register":
        register(model_name, args.experiment, args.run_id)
    elif args.command == "alias":
        set_alias(model_name, checked(args.alias), args.version)
    else:
        list_versions(model_name)


if __name__ == "__main__":
    main()
