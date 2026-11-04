#!/usr/bin/env bash
# 4편 실습 5(선택): 직접 접근 방식 서버(mlflow-direct, practice/ch04/direct-access.yml)로 기록한다.
#
#   bash practice/ch04/direct.sh python practice/ch03/train.py --experiment etth1-ot-6h-direct
#   bash practice/ch04/direct.sh --with-keys python practice/ch03/train.py --experiment etth1-ot-6h-direct
#
# 직접 접근에서는 학습하는 쪽이 S3에 파일을 직접 올린다. 그래서 클라이언트에 세 가지가 필요하다.
#   MLFLOW_S3_ENDPOINT_URL  어느 S3에 올리나 (없으면 boto3는 AWS로 간다)
#   AWS_ACCESS_KEY_ID       S3 키
#   AWS_SECRET_ACCESS_KEY
# --with-keys 없이 돌리면 이 셋이 없는 상태로 돌아서, 모델을 올리는 순간 실패한다.
# workspace 컨테이너에는 점검용 S3_ACCESS_KEY, S3_SECRET_KEY가 들어 있는데, MLflow(boto3)는 AWS_* 이름만 읽는다.
# 그래서 --with-keys일 때만 컨테이너 안에서 AWS_* 이름으로 넘겨 준다.
# (docker-compose.yml의 workspace 환경에 AWS_*나 MLFLOW_S3_ENDPOINT_URL을 추가하면 "키 없이"가 성립하지 않는다)
set -euo pipefail

cd "$(dirname "$0")/../.."

with_keys=false
if [[ "${1:-}" == "--with-keys" ]]; then
  with_keys=true
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "사용법: bash practice/ch04/direct.sh [--with-keys] <명령...>" >&2
  exit 2
fi

# 실패를 빨리 보도록 HTTP 재시도를 줄인다(기본 7회면 몇 분씩 기다린다)
common=(-e MLFLOW_TRACKING_URI=http://mlflow-direct:5002 -e MLFLOW_HTTP_REQUEST_MAX_RETRIES=1)

if [[ "${with_keys}" == "true" ]]; then
  exec docker compose run --rm "${common[@]}" -e MLFLOW_S3_ENDPOINT_URL=http://seaweedfs:8333 workspace \
    bash -c 'export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"; exec "$@"' _ "$@"
fi

exec docker compose run --rm "${common[@]}" workspace "$@"
