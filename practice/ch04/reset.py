"""4편 실습 기록에 삭제 표시를 하고, 완전히 지울 대상의 id를 알려 준다. reset.sh가 부른다.

    bash practice/ch04/reset.sh

지우는 것: 등록 모델 etth1-ot, 실험 etth1-ot-6h와 etth1-ot-6h-direct(그 안의 run과 모델). 대상은 고정이고,
다른 실험은 건드리지 않는다.

MLflow의 삭제는 두 단계다.
1. delete_experiment, delete_logged_model: 삭제 표시만 한다(lifecycle_stage = deleted). 실험을 지우면 그 안의
   run에도 표시된다. DB 행과 버킷의 파일은 그대로 남고, UI의 삭제된 목록에서 되살릴 수 있다. 실험 이름도 계속
   차지해서, 같은 이름으로 set_experiment를 하면 "삭제된 실험"이라며 거부된다.
2. mlflow gc: 삭제 표시된 것을 DB와 버킷에서 실제로 지운다. 되돌릴 수 없다. 서버의 DB 주소와 저장소 키가
   필요해서 reset.sh가 mlflow 컨테이너 안에서 돌린다.
등록 모델은 예외로, delete_registered_model이 바로 지운다(버전과 별칭도 함께). 모델 파일은 LoggedModel 쪽에
있으므로 등록 모델을 지워도 파일은 남는다.

gc의 범위에 주의한다. --run-ids를 주지 않으면 "DB 전체에서 삭제 표시된 모든 run"을, --logged-model-ids를
주지 않으면 "삭제 표시된 모든 모델"을 지운다. --experiment-ids만으로는 범위가 좁혀지지 않는다.
그래서 이 실험들의 run id와 모델 id를 DB에서 읽어 반드시 넘긴다(이미 삭제 표시된 것은 API 검색에 나오지 않는다).
넘길 id가 하나도 없는데 다른 곳에 삭제 표시된 것이 있으면, gc가 그것까지 지우게 되므로 멈춘다.

표준 출력의 마지막 줄에 "<실험 id들>|<run id들>|<모델 id들>"(각각 쉼표 구분)을 쓴다.
"""

import sys
from contextlib import closing
from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.entities import LifecycleStage
from mlflow.exceptions import MlflowException

from practice.ch03.predict import DEFAULT_MODEL_NAME
from practice.ch03.train import EXPERIMENT_NAME
from practice.ch04.inspect_server import connect_db

TARGET_EXPERIMENTS = (EXPERIMENT_NAME, f"{EXPERIMENT_NAME}-direct")

# 표마다: 이 실험들에 속한 행(삭제 표시 여부와 상관없이), 이 실험들 밖에서 삭제 표시된 행 수
TARGET_SQL = {
    "run": "select run_uuid, lifecycle_stage from runs where experiment_id = any(%s)",
    "model": "select model_id, lifecycle_stage from logged_models where experiment_id = any(%s)",
}
OTHER_DELETED_SQL = {
    "run": "select count(*) from runs"
    " where lifecycle_stage = 'deleted' and experiment_id <> all(%s)",
    "model": "select count(*) from logged_models"
    " where lifecycle_stage = 'deleted' and experiment_id <> all(%s)",
}
KIND_LABEL = {"run": "run", "model": "모델"}


@dataclass(frozen=True)
class Targets:
    run_ids: list[str]
    active_model_ids: list[str]
    model_ids: list[str]


def log(message: str) -> None:
    # 표준 출력은 reset.sh가 id를 읽는 데 쓰므로, 사람이 읽을 메시지는 표준 오류로 보낸다
    print(message, file=sys.stderr)


def delete_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.delete_registered_model(name)
        log(f"등록 모델 {name} 삭제(버전, 별칭 포함)")
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        log(f"등록 모델 {name} 없음")


def read_targets(experiment_ids: list[int]) -> Targets:
    """이 실험들의 run과 모델을 DB에서 읽는다. 범위를 좁힐 수 없으면 멈춘다"""
    rows: dict[str, list[tuple[str, str]]] = {}
    with closing(connect_db()) as connection, connection.cursor() as cursor:
        for kind, query in TARGET_SQL.items():
            cursor.execute(query, (experiment_ids,))
            rows[kind] = sorted(cursor.fetchall())
            if rows[kind]:
                continue
            cursor.execute(OTHER_DELETED_SQL[kind], (experiment_ids,))
            (other_deleted,) = cursor.fetchone()
            if other_deleted:
                label = KIND_LABEL[kind]
                raise SystemExit(
                    f"대상 실험에는 {label}이 하나도 없는데, 다른 실험에 삭제 표시된 {label}이"
                    f" {other_deleted}개 있다. gc에 넘길 {label} id가 없으면 gc가 그것까지 지우므로"
                    " 멈춘다. 대상 실험이 비어 있다면 지울 필요 없이 그대로 쓰면 된다"
                )
    return Targets(
        run_ids=[run_id for run_id, _ in rows["run"]],
        active_model_ids=[mid for mid, stage in rows["model"] if stage == LifecycleStage.ACTIVE],
        model_ids=[model_id for model_id, _ in rows["model"]],
    )


def main() -> str:
    client = MlflowClient()
    delete_registered_model(client, DEFAULT_MODEL_NAME)

    experiments = [client.get_experiment_by_name(name) for name in TARGET_EXPERIMENTS]
    found = [experiment for experiment in experiments if experiment is not None]
    log(f"대상 실험: {[f'{e.name}(id {e.experiment_id})' for e in found] or '없음'}")
    if not found:
        print("||")
        return "||"

    targets = read_targets([int(experiment.experiment_id) for experiment in found])
    for model_id in targets.active_model_ids:
        client.delete_logged_model(model_id)
    for experiment in found:
        if experiment.lifecycle_stage == LifecycleStage.ACTIVE:
            client.delete_experiment(experiment.experiment_id)
    log(
        f"삭제 표시: run {len(targets.run_ids)}개, 모델 {len(targets.model_ids)}개(새로 {len(targets.active_model_ids)}개)"
    )

    line = "|".join(
        [
            ",".join(experiment.experiment_id for experiment in found),
            ",".join(targets.run_ids),
            ",".join(targets.model_ids),
        ]
    )
    print(line)
    return line


if __name__ == "__main__":
    main()
