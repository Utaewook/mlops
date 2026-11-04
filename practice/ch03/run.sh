#!/usr/bin/env bash
# 3편 실습용 실행기. workspace 컨테이너에서 명령을 돌리되, 트래킹 서버 없이 로컬에만 기록한다.
#
#   bash practice/ch03/run.sh python practice/ch03/train.py   # 컨테이너에서 명령 실행
#   bash practice/ch03/run.sh bash                            # 컨테이너 셸
#   bash practice/ch03/run.sh ui                              # MLflow UI (http://127.0.0.1:5001)
#
# MLFLOW_TRACKING_URI를 비우면 MLflow는 지금 폴더(/workspace = 저장소 루트)의 mlflow.db(SQLite)에
# run과 레지스트리를 기록한다. --no-deps라서 4편의 PostgreSQL, SeaweedFS, MLflow 서버는 띄우지 않는다.
# UI는 기본 4개 프로세스로 약 1GB를 쓰는데, 혼자 보는 용도라 1개(약 0.4GB)로 줄인다
set -euo pipefail

cd "$(dirname "$0")/../.."
UI_PORT="${MLFLOW_UI_PORT:-5001}"

if [[ $# -eq 0 ]]; then
  echo "사용법: bash practice/ch03/run.sh <명령...> | ui" >&2
  exit 2
fi

if [[ "$1" == "ui" ]]; then
  echo "MLflow UI: http://127.0.0.1:${UI_PORT} (끄려면 Ctrl+C)" >&2
  exec docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= \
    -p "127.0.0.1:${UI_PORT}:${UI_PORT}" \
    workspace mlflow ui --host 0.0.0.0 --port "${UI_PORT}" --workers 1
fi

exec docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= workspace "$@"
