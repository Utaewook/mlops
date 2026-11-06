"""4편: 트래킹 서버 뒤의 두 저장소(PostgreSQL, SeaweedFS)에 무엇이 들어갔는지 직접 열어 본다.

    docker compose run --rm workspace python practice/ch04/inspect_server.py
    docker compose run --rm workspace python practice/ch04/inspect_server.py --experiment etth1-ot-6h-direct

3편의 inspect_store.py와 같은 질문을 서버 구성에 던진다. 3편에서는 SQLite 파일 하나와 mlruns/ 폴더였던 것이
여기서는 PostgreSQL의 표와 S3 버킷의 객체가 된다.

    학습 코드 ──HTTP──▶ MLflow 서버 ──SQL──▶ PostgreSQL   (실험, run, 파라미터, 지표, 모델 행, 레지스트리)
                              └────S3 API──▶ SeaweedFS    (모델 파일. 프록시 방식일 때 서버가 대신 올린다)

학습 코드는 서버 주소(MLFLOW_TRACKING_URI)만 알고 DB 계정과 S3 키는 모른다. 다만 MLflow 3.17은 프록시
방식에서도 모델 파일을 내려받을 때 서버에게 서명된 주소(presigned URL, 5분 유효)만 받고 SeaweedFS에서 직접
받는다. 그래서 키는 없어도 SeaweedFS에 네트워크로 닿아야 한다. 이 스크립트는 점검용이라
PostgreSQL 계정과 S3 키로 두 저장소를 직접 연다(실제 팀이라면 운영자만 가진 권한이다). 읽기만 한다.

두 저장소는 따로 쓰인다. 지표는 PostgreSQL에 들어갔는데 모델 파일은 버킷에 못 올라가는 일이 생길 수 있다.
그래서 마지막에 "DB의 모델 행"과 "버킷의 파일"을 맞대어 본다.
"""

import argparse
import os
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from urllib.parse import urlparse

import boto3
import psycopg2
from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.protos.service_pb2 import LoggedModelStatus
from psycopg2 import sql

from practice.ch03.inspect_store import TABLE_NOTES
from practice.ch03.train import EXPERIMENT_NAME

PROXY_SCHEME = "mlflow-artifacts"
STATUS_PREFIX = "LOGGED_MODEL_"
DB_TIMEOUT_SECONDS = 10

# 실험 하나에서 나온 모델(logged_models)과 그 모델을 만든 run(runs)을 함께 읽는다
MODEL_ROWS_SQL = """
    select m.model_id, coalesce(r.name, '-'), coalesce(r.status, '-'), m.status, m.artifact_location
    from logged_models m
    left join runs r on r.run_uuid = m.source_run_id
    where m.experiment_id = %s
    order by m.creation_timestamp_ms
"""


@dataclass(frozen=True)
class ModelRow:
    model_id: str
    run_name: str
    run_status: str
    status: str
    location: str


def status_name(code: int) -> str:
    """logged_models.status는 숫자로 저장된다. 0 UNSPECIFIED, 1 PENDING, 2 READY, 3 UPLOAD_FAILED"""
    return LoggedModelStatus.Name(code).removeprefix(STATUS_PREFIX)


def artifact_mode(location: str) -> str:
    """실험의 산출물 위치(artifact_location)를 보고 어떤 쓰기 방식인지 말해 준다.

    이 값은 실험을 만들 때 그때의 서버 설정으로 정해져 DB에 저장된다. 나중에 다른 서버로 접속해도 바뀌지 않는다.
    """
    scheme = urlparse(location).scheme
    if scheme == PROXY_SCHEME:
        return (
            "프록시. 올릴 때는 클라이언트가 서버에 HTTP로 보내고 서버가 버킷에 쓴다. 받을 때는 서버가 서명해 준"
            " 주소로 버킷에서 직접 받는다. 어느 쪽이든 클라이언트에 S3 키가 필요 없다"
        )
    if scheme == "s3":
        return "직접 접근. 클라이언트가 이 S3 주소에 직접 쓴다. 클라이언트에 S3 주소와 키가 있어야 한다"
    return "로컬 경로. 이 경로가 보이는 컴퓨터에서만 파일을 열 수 있다(3편)"


def bucket_and_prefix(location: str, proxy_bucket: str) -> tuple[str, str] | None:
    """산출물 위치를 (버킷, 키 접두어)로 바꾼다. S3가 아니면 None.

    mlflow-artifacts:/11/models/m-.../artifacts는 논리 주소다. 실제 버킷은 서버 설정
    (MLFLOW_ARTIFACTS_DESTINATION=s3://<버킷>)이 정하고 클라이언트는 모른다. 여기서는 compose가 서버에 준
    버킷 이름(MLFLOW_BUCKET)을 그 버킷으로 본다.
    """
    parsed = urlparse(location)
    if parsed.scheme == PROXY_SCHEME:
        return proxy_bucket, parsed.path.strip("/")
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.strip("/")
    return None


def verdict(status: str, objects: int | None) -> str:
    """DB의 모델 상태와 버킷의 파일 수가 서로 맞는지 판단한다."""
    if objects is None:
        return "버킷 밖이라 확인하지 않음"
    if status == "READY":
        return "정상" if objects else "DB는 READY인데 버킷에 파일이 없다"
    if status == "UPLOAD_FAILED":
        return f"반쪽 기록: run의 지표는 DB에 있고 모델 파일은 {objects}개"
    if status == "PENDING":
        return "올리는 중이거나, 올리다 멈췄다"
    return "알 수 없는 상태"


def connect_db():
    connection = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ.get("MLFLOW_DB", "mlflow"),
        connect_timeout=DB_TIMEOUT_SECONDS,
    )
    connection.set_session(readonly=True, autocommit=True)
    return connection


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1",
    )


def table_counts(cursor) -> dict[str, int]:
    cursor.execute(
        "select table_name from information_schema.tables"
        " where table_schema = 'public' order by table_name"
    )
    counts = {}
    for (name,) in cursor.fetchall():
        # 표 이름은 DB에서 읽은 값이지만, 그래도 문자열을 이어 붙이지 않고 식별자로 감싼다
        cursor.execute(sql.SQL("select count(*) from {}").format(sql.Identifier(name)))
        counts[name] = cursor.fetchone()[0]
    return counts


def model_rows(cursor, experiment_id: int) -> list[ModelRow]:
    cursor.execute(MODEL_ROWS_SQL, (experiment_id,))
    return [
        ModelRow(model_id, run_name, run_status, status_name(status), location)
        for model_id, run_name, run_status, status, location in cursor.fetchall()
    ]


def list_objects(s3, bucket: str, prefix: str) -> list[tuple[str, int]]:
    paginator = s3.get_paginator("list_objects_v2")
    return [
        (obj["Key"], obj["Size"])
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/")
        for obj in page.get("Contents", [])
    ]


def bucket_model_ids(s3, bucket: str, experiment_prefix: str) -> set[str]:
    """버킷의 <실험>/models/ 아래에 폴더가 있는 모델 id들"""
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{experiment_prefix}/models/", Delimiter="/")
    return {
        common["Prefix"].rstrip("/").rsplit("/", 1)[-1]
        for page in pages
        for common in page.get("CommonPrefixes", [])
    }


def print_connection(experiment: Experiment) -> None:
    client = MlflowClient()
    print("== 1. 클라이언트가 아는 것")
    print(f"트래킹 서버: {client.tracking_uri}")
    print(f"실험: {experiment.name} (id {experiment.experiment_id})")
    print(f"산출물 위치: {experiment.artifact_location}")
    print(f"  → {artifact_mode(experiment.artifact_location)}")


def print_tables(counts: dict[str, int]) -> None:
    print("\n== 2. PostgreSQL: 3편의 mlflow.db와 같은 표가 DB 서버에 있다")
    for name, note in TABLE_NOTES.items():
        if name in counts:
            print(f"{name:<26} {counts[name]:>6}행  {note}")
    others = {name: n for name, n in counts.items() if name not in TABLE_NOTES}
    filled = sum(1 for n in others.values() if n)
    print(f"(그 밖의 표 {len(others)}개 중 행이 있는 표 {filled}개. 대부분 GenAI 기능용)")


def print_models(rows: Sequence[ModelRow], s3, proxy_bucket: str) -> None:
    print("\n== 3. DB의 모델 행과 버킷의 파일 맞대기")
    if not rows:
        print("이 실험에는 모델이 없다")
        return
    for row in rows:
        target = bucket_and_prefix(row.location, proxy_bucket)
        objects = None if target is None else list_objects(s3, *target)
        count = None if objects is None else len(objects)
        print(f"{row.model_id}  run={row.run_name}({row.run_status})  모델={row.status}")
        if target is not None:
            size = sum(size for _, size in objects or [])
            print(f"    s3://{target[0]}/{target[1]}/  파일 {count}개, {size:,}바이트")
        print(f"    → {verdict(row.status, count)}")


def print_orphans(rows: Sequence[ModelRow], s3, experiment: Experiment, proxy_bucket: str) -> None:
    """버킷에는 폴더가 있는데 DB에는 행이 없는 모델. 반대 방향의 어긋남이다"""
    target = bucket_and_prefix(experiment.artifact_location, proxy_bucket)
    if target is None:
        return
    orphans = bucket_model_ids(s3, *target) - {row.model_id for row in rows}
    print(f"\n버킷에만 있고 DB에는 없는 모델 폴더: {sorted(orphans) or '없음'}")


def print_latest_files(rows: Sequence[ModelRow], s3, proxy_bucket: str) -> None:
    ready = [row for row in rows if row.status == "READY"]
    target = bucket_and_prefix(ready[-1].location, proxy_bucket) if ready else None
    if target is None:
        return
    print(f"\n== 4. 가장 최근 READY 모델의 파일 (버킷 {target[0]})")
    for key, size in list_objects(s3, *target):
        print(f"{key}  {size:,}")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PostgreSQL과 SeaweedFS에 들어간 MLflow 기록을 직접 본다"
    )
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="볼 실험 이름")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment = MlflowClient().get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"서버에 실험 {args.experiment!r}이 없다. 먼저 train.py를 서버로 돌린다")
    proxy_bucket = os.environ.get("MLFLOW_BUCKET", "mlflow")

    print_connection(experiment)
    with closing(connect_db()) as connection, connection.cursor() as cursor:
        counts = table_counts(cursor)
        rows = model_rows(cursor, int(experiment.experiment_id))
    print_tables(counts)

    s3 = s3_client()
    print_models(rows, s3, proxy_bucket)
    print_orphans(rows, s3, experiment, proxy_bucket)
    print_latest_files(rows, s3, proxy_bucket)


if __name__ == "__main__":
    main()
