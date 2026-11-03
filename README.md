# MLOps 입문 실습 환경

"MLOps 입문" 블로그 시리즈 3~8편의 실습 환경입니다. 서비스(MLflow, Triton, Dagster 등)와 실습 코드를 돌리는 Python 환경이 모두 컨테이너라서, 어느 PC든 Docker만 있으면 같은 버전으로 재현됩니다.

## 준비물

- Docker Desktop 또는 Docker Engine + Docker Compose (v5.0.2에서 확인했습니다. 오래된 Compose v2에서 `up --wait`가 실패하면 Docker를 업데이트합니다)
- git
- 메모리: 전체 프로필을 띄우면 약 2GB입니다. Docker에 4GB 이상 할당하는 것을 권장합니다
- 디스크: 이미지가 약 24GB입니다. 그중 Triton 이미지가 17GB, 실습용 Python 이미지가 2.7GB입니다. 받은 데이터와 사전학습 모델 캐시가 약 1.5GB 더 듭니다
- GPU는 쓰지 않습니다. 모든 이미지가 amd64와 arm64(Apple Silicon)를 지원합니다

호스트에 Python을 설치할 필요가 없습니다.

## 빠르게 시작

```bash
git clone https://github.com/Utaewook/mlops.git
cd mlops
cp .env.example .env              # 비밀번호와 키를 바꿉니다
docker compose up -d --wait       # 기본 스택(4편)
docker compose run --rm workspace pytest -m base
```

테스트가 통과하면 http://127.0.0.1:5000 에서 MLflow UI가 열립니다.

## 편별 구성

| 편 | 띄우는 명령 | 추가되는 서비스 | 테스트 마커 |
|---|---|---|---|
| 3 | 띄울 것 없음. workspace만 씁니다 ([아래](#실습-코드-돌리기)) | - | - |
| 4 | `docker compose up -d --wait` | postgres, seaweedfs, mlflow | `base` |
| 5 | `docker compose --profile serving up -d --wait` | triton | `serving` |
| 6 | `docker compose --profile orchestration up -d --wait` | dagster-code, dagster-webserver, dagster-daemon | `orchestration` |
| 7 | `docker compose --profile monitoring up -d --wait` | prometheus, grafana | `monitoring` |
| 8 | 4편 스택 그대로. Optuna는 workspace에서 돌립니다 | - | - |

프로필은 겹쳐서 쓸 수 있습니다. 예를 들어 7편에서 Triton 메트릭을 보려면 `--profile serving --profile monitoring`을 같이 줍니다.

## 접속 주소

Windows에서는 `localhost`가 IPv6(::1)로 먼저 풀려서 접속이 안 될 수 있습니다. 그래서 주소를 `127.0.0.1`로 적었습니다. 포트는 `.env`에서 바꿀 수 있습니다.

| 서비스 | 호스트에서 | 컨테이너 안에서 | 인증 |
|---|---|---|---|
| MLflow | http://127.0.0.1:5000 | http://mlflow:5000 | - |
| SeaweedFS S3 | http://127.0.0.1:8333 | http://seaweedfs:8333 | `S3_ACCESS_KEY` / `S3_SECRET_KEY` |
| SeaweedFS 파일 탐색 | http://127.0.0.1:8888 (버킷은 `/buckets` 아래) | - | - |
| PostgreSQL | 127.0.0.1:5432 (DB: `mlflow`, `dagster`) | postgres:5432 | `POSTGRES_USER` / `POSTGRES_PASSWORD` |
| Triton | HTTP 8000, gRPC 8001, 메트릭 8002 | http://triton:8000 | - |
| Dagster | http://127.0.0.1:3000 | http://dagster-webserver:3000 | - |
| Prometheus | http://127.0.0.1:9090 | http://prometheus:9090 | - |
| Grafana | http://127.0.0.1:3001 | http://grafana:3000 | `admin` / `GRAFANA_ADMIN_PASSWORD` |

workspace와 dagster-code 컨테이너에는 `MLFLOW_TRACKING_URI`, `S3_ENDPOINT_URL`, `TRITON_URL` 같은 접속 정보가 환경 변수로 들어 있습니다. 목록은 `docker-compose.yml`의 `x-stack-client-env`에 있습니다.

## 실습 코드 돌리기

`workspace` 컨테이너에는 `uv.lock`으로 고정한 패키지가 설치되어 있고, 저장소가 `/workspace`에 붙어 있습니다. 호스트에서 코드를 고치면 컨테이너에서 바로 돌릴 수 있습니다.

```bash
docker compose run --rm workspace python 파일.py
docker compose run --rm workspace bash          # 셸로 들어가기
```

3편처럼 서버 없이 로컬에만 기록할 때는 트래킹 서버 주소를 비웁니다.

```bash
docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= workspace python 파일.py
docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= -p 127.0.0.1:5001:5001 \
  workspace mlflow ui --host 0.0.0.0 --port 5001
```

6편의 Dagster 코드는 `pipelines/`에 씁니다. dagster-code 컨테이너가 저장소 전체를 읽기 전용으로 붙이므로 3~5편 코드를 import할 수 있습니다. 코드를 고친 뒤에는 Dagster UI에서 Reload를 누릅니다.

Triton 모델 저장소는 `model_repository/`입니다. 5편의 배포 스크립트가 여기에 `모델명/버전/model.onnx`를 넣습니다.

## 실습 과제: Time Series, Vision, LLM

세 유형의 모델을 같은 틀로 다룹니다. 3~8편의 실험 추적, 배포, 파이프라인, 모니터링을 어느 과제에든 똑같이 적용할 수 있도록 했습니다.

| 과제 | 데이터 (라이선스) | 모델 | 슬라이스: 학습 / 검증 / 미래 | 내보내기 |
|---|---|---|---|---|
| `timeseries` | ETTh1 변압기 부하·오일 온도, 1시간 간격 (CC BY-ND 4.0) | LightGBM, 6시간 뒤 오일 온도 (지금 온도에서 변할 양을 학습) | 분기: 2016Q3~2017Q1 / 2017Q2 / 2017Q3~2018Q2 | ONNX |
| `vision` | EuroSAT RGB 위성 사진 10종 (MIT) | MobileNetV3-Small 미세조정 | train / valid / t0(맑음)~t3(짙은 안개) | ONNX |
| `llm` | KLUE YNAT 뉴스 제목 7개 주제 (CC BY-SA 4.0) | Qwen3-0.6B-Base + LoRA, 주제를 생성 | 연도: 2016~2018 / 2019 / 2020 | LoRA 어댑터 |

데이터는 시간 순서가 있는 **슬라이스**로 나눠 `data/<과제>/<슬라이스>.parquet`에 둡니다. 앞쪽 슬라이스로 학습하고, 뒤쪽 "미래" 슬라이스로 성능이 어떻게 변하는지(드리프트) 봅니다. 6편에서는 새 슬라이스가 도착하면 재학습하는 흐름에 그대로 씁니다. 미래 슬라이스에서 생기는 변화는 과제마다 다릅니다.

- `timeseries`: 계절이 바뀌면서 온도와 부하의 관계가 달라집니다
- `vision`: 안개를 합성해서 입력 이미지의 분포가 바뀝니다
- `llm`: 2020년에는 사회 기사 비중이 크게 늘어 라벨 분포가 바뀝니다

```bash
docker compose run --rm --no-deps workspace python -m mlops prepare timeseries   # 내려받고 슬라이스로 나누기
docker compose run --rm --no-deps workspace python -m mlops train timeseries     # 학습, 슬라이스별 평가, 내보내기
docker compose run --rm --no-deps workspace python -m mlops train vision --param epochs=3 --param learning_rate=5e-4
```

`train`은 결과를 `artifacts/<과제>/`에 씁니다. 결과는 임시 폴더에 먼저 쓰고 끝까지 돌았을 때만 바꿔 넣습니다. 그래서 학습이 도중에 죽어도 이전 결과가 그대로 남습니다.

- `report.json`: 파라미터, 슬라이스별 지표, 소요 시간, 최대 메모리
- 내보낸 모델

내보낸 뒤에는 내보낸 파일만으로 다시 예측해서 학습 때 예측과 같은지 확인합니다. 다르면 실패로 끝납니다. 바꿀 수 있는 파라미터는 `python -m mlops train <과제> --param 없는키=1`로 실행하면 오류 메시지에 나옵니다. LLM은 CPU에서 생성이 느려서 슬라이스마다 200행만 평가합니다(`--max-eval-rows`로 바꿉니다).

학습·검증 슬라이스는 실행할 때 바꿀 수 있습니다. 새 슬라이스가 "도착"했다고 보고 학습 구간을 늘려 다시 학습하는 실습(6편)에 씁니다.

```bash
docker compose run --rm --no-deps workspace python -m mlops train timeseries \
  --train-slices 2016Q3,2016Q4,2017Q1,2017Q2,2017Q3 --valid-slices 2017Q4
```

데이터 원본은 커밋이나 리비전 해시로 고정해서 받습니다. 받은 원본과 사전학습 가중치는 `model-cache` 볼륨에 남아 다음부터는 다시 받지 않습니다. `data/`와 `artifacts/`는 git에 올라가지 않습니다. 특히 ETTh1은 CC BY-ND(변경 금지) 라이선스라서 가공한 데이터도 저장소에 올리면 안 됩니다.

**메모리**: 학습 컨테이너(workspace, dagster-code)에는 `TRAINING_MEM_LIMIT`(기본 5g) 상한이 있습니다. 스왑도 같은 값으로 묶어서, 상한을 넘으면 그 컨테이너만 종료됩니다. 세 과제를 동시에 학습하지 말고 하나씩 돌립니다. LLM 학습 중에는 serving, orchestration, monitoring 프로필도 내려 두는 편이 안전합니다.

### 측정 결과

i7-1195G7(4코어), CPU만 쓰고 기본 파라미터로 돌린 결과입니다. 같은 데이터와 시드면 같은 결과가 나옵니다.

| 과제 | 학습 | 평가 | 최대 메모리 | 검증 슬라이스 | 미래 슬라이스 |
|---|---|---|---|---|---|
| `timeseries` | 1초 미만 | 1초 미만 | 0.3GB | 2017Q2 MAE 1.79, 기준선 2.23 (skill 0.20) | skill이 0 근처로 떨어지고 겨울인 2018Q1은 −0.39 |
| `vision` | 1분 24초 | 14초 | 1.1GB | 정확도 0.943 | t0 0.947 → t1 0.768 → t2 0.554 → t3 0.319 |
| `llm` | 18분 | 8분 | 4.3GB | 2019 정확도 0.695, macro-F1 0.647 | 2020 정확도 0.74, macro-F1 0.466 |

- `timeseries`의 skill은 "6시간 뒤에도 지금 온도 그대로"라는 기준선보다 얼마나 나은지(1 − MAE/기준선 MAE)입니다. 검증 슬라이스는 조기 종료에도 쓰므로 조금 낙관적입니다
- `llm`은 2020년에 정확도가 오히려 오릅니다. 라벨 분포가 맞히기 쉬운 사회 기사 쪽으로 쏠렸기 때문입니다. macro-F1은 크게 떨어집니다. 라벨 분포가 바뀔 때는 정확도만 보면 드리프트를 놓칩니다
- `llm`은 슬라이스마다 200행만 평가하므로 정확도에 ±6%p 정도 오차가 있습니다. 시드가 고정이라 재학습 전후 비교는 같은 행으로 합니다

### 과제 코드 구조

```
mlops/
  cli.py          python -m mlops prepare|train
  runner.py       과제와 상관없는 공통 절차: 학습 → 슬라이스별 평가 → 내보내기 → 내보낸 모델 확인
  slices.py       슬라이스 저장과 읽기 (manifest.json에 출처와 행 수)
  download.py     고정된 URL(sha256 확인)과 Hugging Face 리비전으로 내려받기
  tasks/
    base.py       Task 인터페이스
    timeseries.py vision.py llm.py
```

새 과제를 더하려면 `mlops/tasks/<이름>.py`에 `Task` 인터페이스를 따르는 객체를 `TASK`로 두고, `mlops/tasks/__init__.py`의 `TASK_NAMES`에 이름을 넣습니다.

## 의존성 추가

```bash
# pyproject.toml을 고친 뒤
docker compose run --rm --no-deps workspace uv lock
docker compose --profile tools --profile orchestration build
```

`uv.lock`을 커밋하면 다른 PC에서도 같은 버전이 설치됩니다.

## 테스트

`tests/test_stack.py`는 각 서비스가 실제로 연결되는지 확인합니다. 예를 들어 run의 메타데이터는 PostgreSQL에, 산출물은 SeaweedFS에 들어가는지 봅니다. 등록 모델을 별칭으로 불러올 수 있는지, Dagster가 코드를 읽는지, Prometheus가 Triton을 수집하는지도 확인합니다. 테스트가 끝나면 등록 모델과 산출물은 지웁니다. 실험은 MLflow 기본 동작대로 삭제 표시만 되어 UI의 삭제된 실험 목록에 남습니다(`stack-test-*`).

```bash
docker compose run --rm workspace pytest -m base                 # 띄운 프로필만
docker compose run --rm workspace pytest tests/test_stack.py     # 전체 프로필이 떠 있을 때
```

과제 코드의 단위 테스트는 네트워크와 서비스 없이 돕니다. 미래 데이터가 특성에 새지 않는지, 정답 토큰만 학습하는지 같은 것을 확인합니다. 과제별 테스트는 실제 데이터를 내려받아 준비부터 내보내기까지 한 번 돌립니다. 메모리 때문에 하나씩 돌립니다.

```bash
docker compose run --rm --no-deps workspace pytest tests/unit
docker compose run --rm --no-deps workspace pytest -m timeseries
docker compose run --rm --no-deps workspace pytest -m vision
docker compose run --rm --no-deps workspace pytest -m llm        # 작은 테스트용 Qwen3 모델로 구조만 확인
```

## 멈추기와 초기화

```bash
docker compose --profile "*" down        # 컨테이너만 내립니다. 기록은 볼륨에 남습니다
docker compose --profile "*" down -v     # 볼륨까지 지웁니다 (MLflow 기록, 버킷, Dagster 기록)
```

## 고정한 버전

| 구성 요소 | 버전 |
|---|---|
| PostgreSQL | `postgres:18.6` |
| SeaweedFS | `chrislusf/seaweedfs:4.52` |
| MLflow 서버 | `ghcr.io/mlflow/mlflow:v3.17.2` + psycopg2-binary, boto3 |
| Triton | `nvcr.io/nvidia/tritonserver:26.10-py3` |
| Prometheus | `prom/prometheus:v3.15.1` |
| Grafana | `grafana/grafana:13.3.0` |
| Python 환경 | `python:3.12.16-slim-trixie`, uv 0.12.3, 패키지는 `uv.lock` |
| 주요 패키지 | mlflow 3.17.2, dagster 1.13.28, evidently 0.7.23, optuna 5.0.0, scikit-learn 1.9.1, pandas 2.3.3 |
| 학습 | torch 2.15.0+cpu, torchvision 0.29.2+cpu, transformers 5.22.1, peft 0.21.2, lightgbm 4.7.0 |
| 변환·추론 | onnx 1.23.2, onnxmltools 1.16.0, onnxscript 0.7.2, onnxruntime 1.31.1 |
| 데이터·모델 출처 | ETDataset 커밋 `1d16c8f`, `timm/eurosat-rgb` `b4e2855`, `klue/klue` `63647dd`, `Qwen/Qwen3-0.6B-Base` `da87bfb` |

## 구성하며 확인한 것

- MLflow 공식 이미지에는 PostgreSQL 드라이버(psycopg2)와 boto3가 없습니다. 그래서 `infra/mlflow/Dockerfile`에서 추가합니다.
- MLflow 3.x 서버는 Host 헤더 검사가 기본으로 켜져 있어서 localhost와 사설 IP만 허용합니다. 등록되지 않은 Host는 403을 받습니다. 다른 컨테이너가 `http://mlflow:5000`으로 부를 수 있도록 `MLFLOW_SERVER_ALLOWED_HOSTS`에 서비스 이름을 넣었습니다.
- MLflow 3.x 서버는 GenAI 평가용 작업 실행기(huey)도 같이 띄워서 메모리를 약 1.9GB 씁니다. `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`로 끄면 약 0.6GB로 줄어듭니다.
- PostgreSQL 18 이미지부터 데이터 경로가 버전별 하위 폴더로 바뀌었습니다. 그래서 볼륨을 `/var/lib/postgresql`에 붙입니다.
- SeaweedFS의 `mini` 모드는 마스터, 볼륨, 필러, S3를 한 프로세스로 띄웁니다. S3 관리자 키는 `AWS_ACCESS_KEY_ID`와 `AWS_SECRET_ACCESS_KEY` 환경 변수로 받습니다.
- SQLAlchemy 2.1부터 `postgresql://`의 기본 드라이버가 psycopg(v3)로 바뀌었습니다. 그런데 dagster-postgres는 psycopg2만 설치하기 때문에 Dagster가 뜨지 않았습니다. 그래서 `dagster.yaml`에 `scheme: postgresql+psycopg2`를 적었습니다.
- evidently 0.7.23은 pandas 3에서 import가 깨집니다. 그래서 `pandas<3`으로 고정했습니다.
- PyPI의 기본 torch 휠은 CUDA 라이브러리까지 받습니다. 그래서 `pyproject.toml`에서 PyTorch CPU 인덱스를 지정했습니다.
- LightGBM은 OpenMP 런타임(`libgomp1`)이 필요한데 slim 이미지에는 없습니다. 그래서 Dockerfile에서 설치합니다.
- ETTh1으로 24시간 뒤 온도를 예측해 보니, 어떤 설정도 "지금 온도 그대로"라는 기준선을 넘지 못했습니다(검증 MAE 2.42 대 2.46). 24시간 뒤가 하루 주기와 겹치기 때문입니다. 그래서 6시간 뒤를 예측합니다(1.79 대 2.23). 또 트리 모델은 학습 때 본 범위 밖의 값을 내지 못해서, 온도 자체가 아니라 지금 온도에서 얼마나 변할지를 학습합니다.
- onnxmltools 1.16의 LightGBM 변환기는 opset 15까지만 지원합니다. 변환된 그래프에 "지금 온도 + 변화량" 덧셈 노드를 붙여서 ONNX 모델이 예측 온도를 바로 내놓게 했습니다.
- `torch.onnx.export`는 기본으로 가중치를 별도 파일(`model.onnx.data`)로 저장합니다. 작은 모델이라 `external_data=False`로 한 파일에 담았습니다.
- KLUE YNAT의 날짜는 `2016.06.30. 오후 3:07`과 `2018.06.04 08:15` 두 형식이 섞여 있습니다.
- PyTorch를 CPU에서 돌리면 glibc malloc이 해제한 텐서 메모리를 OS에 돌려주지 않고 힙에 남겨 둡니다. LLM 학습 중 RSS가 4.8GB였고, 학습한 모델을 놓고 내보낸 모델을 다시 올리다가 5GB 상한에 걸려 종료됐습니다. Dockerfile에서 `MALLOC_ARENA_MAX=2`와 `MALLOC_MMAP_THRESHOLD_=1048576`을 주면 학습 중 2.8GB, 최대 4.1GB로 줄고 학습 속도는 그대로였습니다.
- pandas 2.3과 NumPy 2.5 조합에서는 `pd.Timedelta(hours=1)`이 DeprecationWarning을 냅니다. `pd.Timedelta(1, unit="h")`로 씁니다.

## 문제 해결

- 포트가 이미 쓰이고 있으면 `.env`의 `*_PORT` 값을 바꿉니다. macOS는 AirPlay 수신이 5000번을 씁니다.
- `.env`를 만들기 전에 compose 명령을 실행하면 ".env를 만드세요"라는 오류가 납니다.
- PostgreSQL 초기화 스크립트(`infra/postgres/init`)는 볼륨이 비어 있을 때 한 번만 돕니다. DB 이름이나 계정을 바꿨다면 `down -v`로 볼륨을 지우고 다시 띄웁니다.
- workspace 컨테이너는 `.env`의 `HOST_UID`/`HOST_GID`(기본 1000) 사용자로 돕니다. Linux에서 `id -u`가 1000이 아니면 이 값을 바꿔야 합니다. 그래야 컨테이너가 만든 파일(`mlflow.db` 등)을 호스트에서 고치거나 지울 수 있습니다.
