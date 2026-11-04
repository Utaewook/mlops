"""4편 점검 스크립트의 판단 로직. 실제 DB와 버킷 대조는 스택을 띄운 뒤 스크립트로 확인한다."""

import pytest

from practice.ch04.inspect_server import artifact_mode, bucket_and_prefix, status_name, verdict


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        # 프록시: 논리 주소의 경로가 서버 버킷의 키가 된다
        ("mlflow-artifacts:/11/models/m-1/artifacts", ("mlflow", "11/models/m-1/artifacts")),
        ("mlflow-artifacts:/11", ("mlflow", "11")),
        # 직접 접근: 주소에 버킷이 들어 있다
        ("s3://other/direct/12/models/m-2/artifacts", ("other", "direct/12/models/m-2/artifacts")),
        # 3편의 로컬 경로는 버킷이 아니다
        ("/workspace/mlruns/1", None),
    ],
)
def test_bucket_and_prefix(location, expected):
    assert bucket_and_prefix(location, proxy_bucket="mlflow") == expected


def test_artifact_mode_names_who_needs_keys():
    assert "S3 키가 필요 없다" in artifact_mode("mlflow-artifacts:/11")
    assert "키가 있어야 한다" in artifact_mode("s3://mlflow/direct/12")
    assert "로컬 경로" in artifact_mode("/workspace/mlruns/1")


def test_status_name_reads_db_codes():
    assert [status_name(code) for code in (1, 2, 3)] == ["PENDING", "READY", "UPLOAD_FAILED"]


@pytest.mark.parametrize(
    ("status", "objects", "expected"),
    [
        ("READY", 7, "정상"),
        ("READY", 0, "DB는 READY인데 버킷에 파일이 없다"),
        ("UPLOAD_FAILED", 0, "반쪽 기록: run의 지표는 DB에 있고 모델 파일은 0개"),
        ("PENDING", 2, "올리는 중이거나, 올리다 멈췄다"),
        ("READY", None, "버킷 밖이라 확인하지 않음"),
    ],
)
def test_verdict(status, objects, expected):
    assert verdict(status, objects) == expected
