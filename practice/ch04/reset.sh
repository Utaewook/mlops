#!/usr/bin/env bash
# 4편 실습 기록(실험 etth1-ot-6h, etth1-ot-6h-direct와 등록 모델 etth1-ot)을 DB와 버킷에서 완전히 지운다.
# 되돌릴 수 없다. 다른 실험과 스택 테스트 기록은 건드리지 않는다.
#
#   bash practice/ch04/reset.sh
#
# 1단계(workspace): MLflow API로 삭제 표시를 하고, 지울 실험, run, 모델의 id를 받는다 (reset.py)
# 2단계(mlflow 서버 컨테이너): mlflow gc로 그 id들만 DB 행과 버킷 파일까지 지운다.
#   gc는 HTTP API가 아니라 DB에 직접 붙어서 지우므로, DB 주소(MLFLOW_BACKEND_STORE_URI)와
#   저장소 키를 가진 서버 컨테이너에서 돌린다.
#   gc는 --run-ids, --logged-model-ids를 빼면 다른 실험의 삭제 표시된 것까지 지운다. 그래서 id를 늘 넘긴다
set -euo pipefail

cd "$(dirname "$0")/../.."

ids="$(docker compose run --rm -T workspace python practice/ch04/reset.py | tail -n 1)"
# 기대하는 모양: "<실험 id들>|<run id들>|<모델 id들>". 다른 것이 섞였으면 gc에 넘기지 않는다
if [[ ! "${ids}" =~ ^[0-9,]*\|[0-9a-f,]*\|[A-Za-z0-9_,-]*$ ]]; then
  echo "reset.py 출력이 예상과 다르다: ${ids}" >&2
  exit 1
fi
IFS='|' read -r experiment_ids run_ids model_ids <<< "${ids}"

if [[ -z "${experiment_ids}" ]]; then
  echo "지울 실험이 없다" >&2
  exit 0
fi

gc_args=(--experiment-ids "${experiment_ids}")
# 빈 목록이면 reset.py가 "다른 곳에도 삭제 표시된 것이 없음"을 확인했으므로 빼도 범위가 넓어지지 않는다
if [[ -n "${run_ids}" ]]; then
  gc_args+=(--run-ids "${run_ids}")
fi
if [[ -n "${model_ids}" ]]; then
  gc_args+=(--logged-model-ids "${model_ids}")
fi

# --artifacts-destination: mlflow-artifacts:/ 주소를 실제 버킷으로 풀 때 쓴다(서버 설정과 같은 값)
docker compose exec -T mlflow sh -c \
  'mlflow gc --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" --tracking-uri http://127.0.0.1:5000 \
     --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" "$@"' _ "${gc_args[@]}"
