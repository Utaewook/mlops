"""실습 스택 통합 테스트. workspace 컨테이너 안에서 서비스 이름으로 접속한다.

띄운 프로필에 맞는 마커로 실행한다.

    docker compose up -d
    docker compose run --rm workspace pytest -m base

    docker compose --profile serving --profile orchestration --profile monitoring up -d
    docker compose run --rm workspace pytest
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import psycopg2
import pytest
import requests
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression

MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_URL = os.environ.get("S3_ENDPOINT_URL", "http://seaweedfs:8333")
TRITON_URL = os.environ.get("TRITON_URL", "http://triton:8000")
TRITON_METRICS_URL = os.environ.get("TRITON_METRICS_URL", "http://triton:8002")
DAGSTER_URL = os.environ.get("DAGSTER_URL", "http://dagster-webserver:3000")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://grafana:3000")

MLFLOW_BUCKET = os.environ.get("MLFLOW_BUCKET", "mlflow")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "data")
REQUEST_TIMEOUT_SECONDS = 10
DAEMON_STARTUP_TIMEOUT_SECONDS = 120


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} 환경 변수가 없습니다. `docker compose run --rm workspace pytest`로 실행하세요.")
    return value


def http_get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)


def wait_until(check: Callable[[], bool], timeout: float, interval: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(interval)
    return False


def query_mlflow_db(sql: str, params: tuple[str, ...]) -> list[tuple]:
    """MLflow 메타데이터 DB(PostgreSQL)에 파라미터 쿼리를 보낸다."""
    connection = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        user=require_env("POSTGRES_USER"),
        password=require_env("POSTGRES_PASSWORD"),
        dbname=os.environ.get("MLFLOW_DB", "mlflow"),
        connect_timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        connection.close()


def delete_prefix(s3, bucket: str, prefix: str) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})


# ---------- base: PostgreSQL + SeaweedFS + MLflow (4편) ----------


@dataclass(frozen=True)
class ExperimentRef:
    id: str
    name: str


@pytest.fixture(scope="module")
def s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_URL,
        aws_access_key_id=require_env("S3_ACCESS_KEY"),
        aws_secret_access_key=require_env("S3_SECRET_KEY"),
        region_name="us-east-1",
    )


@pytest.fixture
def experiment(s3) -> Iterator[ExperimentRef]:
    """테스트 전용 실험을 만들고, 끝나면 등록 모델, 실험, 산출물을 지운다."""
    mlflow.set_tracking_uri(MLFLOW_URL)
    client = MlflowClient()
    name = f"stack-test-{uuid.uuid4().hex[:8]}"
    created = ExperimentRef(id=client.create_experiment(name), name=name)
    yield created
    if client.search_registered_models(filter_string=f"name = '{name}'"):
        client.delete_registered_model(name)
    client.delete_experiment(created.id)
    delete_prefix(s3, MLFLOW_BUCKET, f"{created.id}/")


@pytest.mark.base
def test_mlflow_server_is_healthy():
    response = http_get(f"{MLFLOW_URL}/health")

    assert response.status_code == 200


@pytest.mark.base
def test_object_store_has_mlflow_and_data_buckets(s3):
    bucket_names = {bucket["Name"] for bucket in s3.list_buckets()["Buckets"]}

    assert {MLFLOW_BUCKET, DATA_BUCKET} <= bucket_names


@pytest.mark.base
def test_run_metadata_goes_to_postgres_and_artifacts_to_object_store(experiment, s3):
    with mlflow.start_run(experiment_id=experiment.id) as run:
        mlflow.log_param("alpha", "0.5")
        mlflow.log_metric("rmse", 1.23)
        mlflow.log_text("hello", "notes/hello.txt")

    # 클라이언트는 S3 키 없이 트래킹 서버에만 보내고, 서버가 저장소에 쓴다(프록시 방식)
    assert run.info.artifact_uri.startswith("mlflow-artifacts:")
    key = f"{experiment.id}/{run.info.run_id}/artifacts/notes/hello.txt"
    assert s3.get_object(Bucket=MLFLOW_BUCKET, Key=key)["Body"].read() == b"hello"
    stored_params = query_mlflow_db(
        "SELECT key, value FROM params WHERE run_uuid = %s", (run.info.run_id,)
    )
    assert stored_params == [("alpha", "0.5")]


@pytest.mark.base
def test_registered_model_loads_by_alias(experiment):
    features = np.arange(10, dtype=float).reshape(-1, 1)
    model = LinearRegression().fit(features, 2 * features.ravel() + 1)
    with mlflow.start_run(experiment_id=experiment.id):
        model_info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=experiment.name, input_example=features[:2]
        )

    MlflowClient().set_registered_model_alias(
        experiment.name, "champion", model_info.registered_model_version
    )
    loaded = mlflow.pyfunc.load_model(f"models:/{experiment.name}@champion")

    np.testing.assert_allclose(loaded.predict(features[:3]), model.predict(features[:3]))


# ---------- serving: Triton (5편) ----------


@pytest.mark.serving
def test_triton_is_ready():
    response = http_get(f"{TRITON_URL}/v2/health/ready")

    assert response.status_code == 200


@pytest.mark.serving
def test_triton_model_repository_api_responds():
    response = requests.post(f"{TRITON_URL}/v2/repository/index", timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.serving
def test_triton_exposes_prometheus_metrics():
    response = http_get(f"{TRITON_METRICS_URL}/metrics")

    assert response.status_code == 200
    assert "nv_" in response.text


# ---------- orchestration: Dagster (6편) ----------


def dagster_query(query: str) -> dict:
    response = requests.post(
        f"{DAGSTER_URL}/graphql", json={"query": query}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    body = response.json()
    assert "errors" not in body, body
    return body["data"]


@pytest.mark.orchestration
def test_dagster_loads_pipelines_code_location():
    query = (
        "{ workspaceOrError { ... on Workspace {"
        " locationEntries { name locationOrLoadError { __typename } } } } }"
    )

    def pipelines_location_loaded() -> bool:
        entries = dagster_query(query)["workspaceOrError"]["locationEntries"]
        load_states = {
            entry["name"]: (entry["locationOrLoadError"] or {}).get("__typename")
            for entry in entries
        }
        return load_states.get("pipelines") == "RepositoryLocation"

    # webserver가 healthy여도 코드 위치는 비동기로 읽으므로 기다린다
    assert wait_until(pipelines_location_loaded, timeout=DAEMON_STARTUP_TIMEOUT_SECONDS)


@pytest.mark.orchestration
def test_dagster_required_daemons_are_healthy():
    query = "{ instance { daemonHealth { allDaemonStatuses { daemonType required healthy } } } }"

    def required_daemons_healthy() -> bool:
        statuses = dagster_query(query)["instance"]["daemonHealth"]["allDaemonStatuses"]
        return all(status["healthy"] is True for status in statuses if status["required"])

    assert wait_until(required_daemons_healthy, timeout=DAEMON_STARTUP_TIMEOUT_SECONDS)


# ---------- monitoring: Prometheus + Grafana (7편) ----------


@pytest.mark.monitoring
def test_prometheus_is_ready():
    response = http_get(f"{PROMETHEUS_URL}/-/ready")

    assert response.status_code == 200


def prometheus_targets() -> dict[str, dict]:
    targets = http_get(f"{PROMETHEUS_URL}/api/v1/targets").json()["data"]["activeTargets"]
    return {target["labels"]["job"]: target for target in targets}


def triton_is_reachable() -> bool:
    try:
        return http_get(f"{TRITON_URL}/v2/health/live").status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.monitoring
def test_prometheus_has_triton_scrape_target():
    triton_target = prometheus_targets().get("triton")

    assert triton_target is not None
    assert triton_target["scrapeUrl"] == "http://triton:8002/metrics"


@pytest.mark.monitoring
@pytest.mark.serving
def test_prometheus_scrapes_running_triton():
    if not triton_is_reachable():
        pytest.skip("Triton이 떠 있지 않다. serving 프로필을 같이 띄우면 확인한다")

    def triton_target_up() -> bool:
        return prometheus_targets().get("triton", {}).get("health") == "up"

    # 첫 수집은 scrape_interval(15초)만큼 걸릴 수 있다
    assert wait_until(triton_target_up, timeout=60)


@pytest.mark.monitoring
def test_grafana_reaches_prometheus_datasource():
    auth = ("admin", require_env("GRAFANA_ADMIN_PASSWORD"))

    response = http_get(f"{GRAFANA_URL}/api/datasources/uid/prometheus/health", auth=auth)

    assert response.status_code == 200
    assert response.json()["status"] == "OK"
