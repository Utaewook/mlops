# 4편 실습: 기반 스택 (트래킹 서버 + PostgreSQL + SeaweedFS)

"MLOps 입문" 4편의 실습서입니다. 3편에서 로컬(`mlflow.db`, `mlruns/`)에 남기던 기록을 트래킹 서버로 보냅니다. 학습과 레지스트리 코드는 3편의 `practice/ch03/` 것을 **한 줄도 바꾸지 않고** 그대로 씁니다.

진행 방식은 3편과 같습니다. 단계마다 예측 → 실행 → 코드가 지나가는 길 → 관찰(캡처 포함) → 원리 순서로 갑니다. 3편 1단계에서 붙인 번호(①~⑩, ⑧-1~⑧-6)를 그대로 씁니다. **3편 1단계를 먼저 읽고 오세요.**

3편에서는 MLflow 함수가 같은 프로세스 안에서 SQLite 파일에 바로 썼습니다. 그래서 "코드가 지나가는 길"을 소스를 읽어서만 따라갈 수 있었습니다. 4편에서는 같은 함수 호출이 하나하나 **서버에 보내는 HTTP 요청**이 됩니다. 그 길이 서버 로그에 그대로 찍힙니다. 이 편에서 가장 많이 쓰는 관찰 도구가 이 로그입니다. 터미널 하나를 로그 전용으로 띄워 둡니다(터미널 A).

```bash
# /health는 healthcheck가 5초마다 부르는 요청이라 뺀다. --line-buffered가 없으면 grep이 출력을 모아 두었다가 늦게 내보낸다
docker compose logs -f --no-log-prefix mlflow | grep --line-buffered -v /health
```

코드 위치는 `파일:줄`로 적었습니다. 서버 쪽 MLflow 코드는 mlflow 컨테이너의 `/usr/local/lib/python3.11/site-packages/mlflow/`에, 클라이언트 쪽은 workspace 컨테이너의 `/opt/venv/lib/python3.12/site-packages/mlflow/`에 있습니다. 두 버전은 같습니다(3.17.2).

```bash
# 예: 서버의 /health가 하는 일. 1단계와 4단계에서 이 두 줄이 중요해진다
docker compose exec mlflow sed -n 105,106p /usr/local/lib/python3.11/site-packages/mlflow/server/__init__.py
```

"이 PC에서 본 값"은 2026-11-04~05에 확인한 값입니다. 서버 요청 로그는 같은 compose 파일을 다른 프로젝트 이름(`-p mlops-lab04`)과 다른 포트로 띄운 **빈 스택**에서 받았습니다. 그래서 실험 id가 1입니다. 여러분의 스택에는 스택 테스트가 먼저 만든 실험이 있어서 id가 더 큽니다. 아래에서 `<실험 id>`로 적은 곳에는 자기 값을 넣습니다.

## 전체 그림

```text
                        기록(②~⑨): HTTP          ┌── SQL ──────▶ PostgreSQL  DB "mlflow": 3편 mlflow.db와 같은 표
train.py ─────────────────────────▶ MLflow 서버 ─┤
(workspace)   모델 파일 올리기: HTTP PUT          └── S3 API ───▶ SeaweedFS   버킷 "mlflow": <실험 id>/models/m-…/artifacts/…
                                    (mlflow:5000)                    ▲
predict.py ── 서명된 주소 받기: HTTP ──▶ 서버                         │
(workspace) └─ 파일 받기: 그 주소로 버킷에 직접 ───────────────────────┘

아는 것   학습·예측 쪽: 서버 주소 하나      서버: DB 계정, S3 주소와 키, 버킷 이름
```

학습하는 쪽은 서버 주소(`MLFLOW_TRACKING_URI=http://mlflow:5000`)만 압니다. 모델 파일을 **올릴 때**는 서버가 받아서 대신 버킷에 씁니다. 노트의 **프록시 방식**이 이것입니다. 그런데 **내려받을 때**는 길이 다릅니다. 서버는 "5분 동안 이 파일을 받아도 된다"는 서명이 붙은 주소만 주고, 파일은 클라이언트가 버킷에서 직접 받습니다. 노트의 "산출물 쓰기 방식" 표에는 없는 부분이고, 2-4에서 확인합니다.

## 준비

```bash
docker compose up -d --wait                      # postgres, seaweedfs, create-buckets, mlflow
docker compose ps                                # 상태가 healthy인지
docker compose run --rm workspace pytest -m base # 서비스끼리 실제로 연결되는지 (스택 테스트)
```

### 코드가 지나가는 길

- `up`은 profile이 없는 서비스만 띄웁니다. 그래서 postgres, seaweedfs, create-buckets, mlflow 넷입니다. triton(5편), dagster(6편), prometheus·grafana(7편)는 profile이 있어서 뜨지 않습니다([`docker-compose.yml:3-7`](../../docker-compose.yml#L3-L7)). workspace도 `profiles: [tools]`([`docker-compose.yml:272`](../../docker-compose.yml#L272))라서 뜨지 않고, `docker compose run`으로 그때그때 만들어 씁니다.
- `--wait`는 서비스마다 healthcheck가 healthy가 될 때까지, create-buckets처럼 일하고 끝나는 컨테이너는 끝날 때까지 기다립니다. 띄우는 순서는 1단계에서 봅니다.
- `pytest -m base`는 [`tests/test_stack.py`](../../tests/test_stack.py)에서 `base` 표시가 붙은 테스트 4개를 돌립니다. 서버가 응답하는지, 버킷 두 개가 있는지, run 기록이 PostgreSQL로 가고 파일이 버킷으로 가는지, 등록한 모델을 별칭으로 불러올 수 있는지 봅니다. 이 테스트가 실험을 하나 만들기 때문에 여러분의 `etth1-ot-6h` 실험 id가 1이 아닙니다.

### 관찰

이 PC의 격리 스택에서 본 `ps`입니다(포트만 1만 번대로 바꿨습니다).

```text
NAME                      IMAGE                      SERVICE     STATUS                    PORTS
mlops-lab04-mlflow-1      mlops/mlflow:3.17.2        mlflow      Up 11 seconds (healthy)   127.0.0.1:15000->5000/tcp
mlops-lab04-postgres-1    postgres:18.6              postgres    Up 22 seconds (healthy)   127.0.0.1:15432->5432/tcp
mlops-lab04-seaweedfs-1   chrislusf/seaweedfs:4.52   seaweedfs   Up 22 seconds (healthy)   …, 127.0.0.1:18333->8333/tcp, 127.0.0.1:18888->8888/tcp
```

- mlflow만 Up 시간이 짧습니다. postgres와 seaweedfs가 healthy가 된 뒤에야 시작했기 때문입니다.
- 포트가 모두 `127.0.0.1:`로 시작합니다. 이 PC 밖에서는 접속할 수 없습니다.
- create-buckets가 없습니다. `docker compose ps -a`로 보면 `Exited (0)`입니다. 버킷을 만들고 정상으로 끝난 것입니다.

**캡처 `mlops-compose-ps.png`**: `docker compose ps` 결과를 찍습니다. healthy 세 개와 `127.0.0.1` 포트가 함께 보이면 됩니다. 이 화면은 "세 서비스가 떠 있고, 각자 스스로 건강하다고 말하고 있다"는 것까지만 보여 줍니다. 서로 연결되어 일하는지는 이 화면으로 알 수 없고, 4단계에서 그 차이를 봅니다. 노트 04의 `<!-- 실습 1 (compose로 스택 띄우기)` 주석 안 `캡처:` 항목에 적힌 그림입니다. 주석을 풀어 본문을 쓸 때 서비스별 설정 이유 표 다음에 넣습니다. 노트에 적혀 있듯 코드 블록으로 대신해도 됩니다.

데이터는 3편에서 준비한 `data/timeseries/`를 그대로 씁니다. 처음부터 다시 하려면 `bash practice/ch04/reset.sh`로 4편 기록을 지웁니다(맨 아래 [처음부터 다시 하기](#처음부터-다시-하기)).

## 1. 스택 읽기: 누가 무엇을 아는가

> 목표: compose 파일의 줄마다 그 값을 누가 읽고 무엇이 정해지는지 설명할 수 있다.

**예측**: MLflow 서버를 띄우는 명령에는 DB 주소가 없습니다. 서버는 DB 주소를 어디서 알까요? 학습 컨테이너(workspace)에는 S3 키가 있을까요, 없을까요?

### 코드가 지나가는 길

**mlflow 서비스**: [`docker-compose.yml:123-153`](../../docker-compose.yml#L123-L153)

```yaml
command: ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]                       # 127
environment:
  MLFLOW_BACKEND_STORE_URI: postgresql+psycopg2://${POSTGRES_USER}:…@postgres:5432/mlflow  # 129
  MLFLOW_ARTIFACTS_DESTINATION: s3://${MLFLOW_BUCKET:-mlflow}                              # 133
  MLFLOW_S3_ENDPOINT_URL: http://seaweedfs:8333                                           # 134
  AWS_ACCESS_KEY_ID: ${S3_ACCESS_KEY}                                                     # 135
  AWS_SECRET_ACCESS_KEY: ${S3_SECRET_KEY}                                                 # 136
  MLFLOW_WORKERS: "2"                                                                     # 137
```

명령에는 `--host`, `--port`밖에 없습니다. `mlflow server`의 옵션은 저마다 같은 뜻의 환경 변수 이름을 함께 가지고 있어서, 옵션 대신 환경 변수로 줘도 됩니다. MLflow 소스에서 그 짝을 확인할 수 있습니다.

```python
# mlflow/cli/__init__.py:371-373
@click.option(
    "--backend-store-uri",
    envvar="MLFLOW_BACKEND_STORE_URI",
```

| 환경 변수 | 읽는 쪽 | 정해지는 것 |
|---|---|---|
| `MLFLOW_BACKEND_STORE_URI` | `mlflow server`(`cli/__init__.py:373`) | 메타데이터를 어느 DB에 쓸지. `postgresql+psycopg2`의 `+psycopg2`는 SQLAlchemy가 쓸 드라이버 이름이다. 공식 이미지에 이 드라이버가 없어서 [`infra/mlflow/Dockerfile`](../../infra/mlflow/Dockerfile)에서 추가한다 |
| `MLFLOW_ARTIFACTS_DESTINATION` | `mlflow server`(`cli/__init__.py:1075`) | 파일을 실제로 둘 곳. 새 실험의 산출물 위치는 이 값이 아니라 `mlflow-artifacts:/<실험 id>`라는 논리 주소로 저장되고, 서버가 그 주소를 이 버킷으로 풀어서 쓴다 |
| `MLFLOW_S3_ENDPOINT_URL` | 서버 안의 boto3 | 어느 S3에 붙을지. 없으면 boto3는 진짜 AWS로 간다 |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | 서버 안의 boto3 | S3 키. boto3는 이 이름만 읽는다 |
| `MLFLOW_WORKERS` | `mlflow server`(`mlflow/utils/cli_args.py:195`) | 요청을 처리할 프로세스 수. 기본 4개인데 메모리를 아끼려고 2개로 줄였다 |

프록시 방식을 켜는 옵션(`--serve-artifacts`)은 따로 주지 않았습니다. MLflow 3.x에서는 기본값이 켜짐(`Default: True`)입니다. 끄는 쪽은 5단계에서 해 봅니다.

**workspace 서비스**: [`docker-compose.yml:269-292`](../../docker-compose.yml#L269-L292)는 환경 변수를 [`docker-compose.yml:26-38`](../../docker-compose.yml#L26-L38)의 묶음에서 받습니다.

```yaml
MLFLOW_TRACKING_URI: http://mlflow:5000      # 28: MLflow 클라이언트가 읽는다
S3_ENDPOINT_URL: http://seaweedfs:8333       # 29
S3_ACCESS_KEY: ${S3_ACCESS_KEY}              # 30: 이름이 AWS_*가 아니다
S3_SECRET_KEY: ${S3_SECRET_KEY}              # 31
```

S3 키가 있긴 하지만 이름이 `S3_*`입니다. MLflow와 boto3는 이 이름을 읽지 않으므로, 학습 코드 입장에서는 키가 없는 것과 같습니다. 이 값은 2-2의 점검 스크립트(`inspect_server.py`)와 스택 테스트가 저장소를 직접 열어 볼 때만 씁니다. 5단계의 `direct.sh --with-keys`는 이 값을 일부러 `AWS_*` 이름으로 바꿔서 넘깁니다.

**띄우는 순서**: `depends_on`의 조건이 사슬을 이룹니다.

```text
postgres  healthy (79: pg_isready)  ──────────────────────────────────┐
seaweedfs healthy (103: /healthz) ─▶ create-buckets 정상 종료 (118-121) ┴─▶ mlflow healthy (152: /health) ─▶ workspace (290-292)
```

- create-buckets는 [`infra/seaweedfs/create_buckets.py`](../../infra/seaweedfs/create_buckets.py)를 돌립니다. 27줄에서 버킷을 만들고, 이미 있으면 30줄에서 오류 코드를 보고 건너뜁니다. 그래서 몇 번을 다시 돌려도 결과가 같습니다.
- postgres의 healthcheck가 `-h 127.0.0.1`로 TCP를 쓰는 이유는 79줄 위의 주석에 있습니다. 첫 초기화 중에 뜨는 임시 서버는 소켓만 열기 때문에, 소켓으로 물으면 초기화가 끝나기 전에 healthy가 됩니다.
- mlflow의 healthcheck(152줄)는 `/health`를 부릅니다. 서버 쪽 코드는 이것이 전부입니다.

```python
# mlflow/server/__init__.py:105-106
def health():
    return "OK", 200
```

DB도 버킷도 건드리지 않습니다. 4단계에서 이 두 줄의 의미가 드러납니다.

### 관찰

값에는 비밀번호와 키가 들어 있으므로 이름만 봅니다(`cut -d= -f1`).

```bash
docker compose exec mlflow sh -c 'env | cut -d= -f1 | grep -E "^(MLFLOW|AWS)_" | sort'
docker compose run --rm --no-deps workspace sh -c 'env | cut -d= -f1 | grep -E "^(MLFLOW|AWS|S3)_" | sort'
curl -s http://127.0.0.1:5000/health; echo
curl -s http://127.0.0.1:5000/api/3.0/mlflow/server-info; echo
```

이 PC에서 본 출력입니다.

| mlflow 컨테이너 | workspace 컨테이너 |
|---|---|
| `AWS_ACCESS_KEY_ID` | |
| `AWS_SECRET_ACCESS_KEY` | |
| `MLFLOW_ARTIFACTS_DESTINATION` | |
| `MLFLOW_BACKEND_STORE_URI` | |
| `MLFLOW_S3_ENDPOINT_URL` | |
| | `MLFLOW_TRACKING_URI` |
| | `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_ENDPOINT_URL` (점검용) |
| `MLFLOW_SERVER_ALLOWED_HOSTS`, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION`, `MLFLOW_WORKERS` | `MLFLOW_BUCKET`, `MLFLOW_DB`, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION` |

`/health`는 `OK`, `server-info`는 이렇게 나옵니다(줄여서 옮겼습니다).

```json
{"store_type": "SqlStore", "multipart_uploads_enabled": true, "multipart_downloads_enabled": true, ...}
```

`multipart_downloads_enabled: true`를 기억해 둡니다. 2-4에서 모델을 내려받는 길을 바꾸는 값입니다.

### 원리

- **비밀은 서버 한 곳에만 있습니다.** DB 계정과 S3 키는 mlflow 컨테이너에만 있고, 학습하는 쪽은 주소 하나만 압니다. 키를 바꿀 때 고칠 곳도 서버 하나입니다.
- **어디에 저장할지는 서버 설정이 정합니다.** 클라이언트는 `mlflow-artifacts:/…`라는 논리 주소만 보고, 그것이 어느 버킷인지 모릅니다. 그래서 버킷을 옮겨도 학습 코드는 바뀌지 않습니다.
- **`depends_on`의 조건은 시작할 때 한 번만 봅니다.** mlflow가 뜬 뒤에 postgres가 죽어도 compose는 mlflow를 멈추지 않습니다. 그리고 `/health`는 DB와 버킷을 보지 않으므로 mlflow는 계속 healthy입니다. 4단계에서 직접 확인합니다.

구성하다 막혔던 부분(증상과 해결)은 저장소 루트 README의 "구성하며 확인한 것"에 있습니다. MLflow 3.x 서버의 Host 헤더 검사(142줄의 `MLFLOW_SERVER_ALLOWED_HOSTS`), 작업 실행기의 메모리 사용(140줄), PostgreSQL 18의 데이터 경로(71-72줄) 같은 것들입니다.

## 2. 3편 실습을 서버로 옮기기

### 2-1. 같은 코드로 학습하기: 요청 로그로 ①~⑩ 다시 따라가기

> 목표: 코드에서 바뀌는 것이 정말 주소 하나뿐인지 확인하고, 3편에서 소스로만 따라간 길을 서버 로그로 눈으로 본다.

**예측**: run 하나에 HTTP 요청이 몇 개나 갈까요? 모델 파일 7개는 어떤 요청으로 갈까요? 서버 로그가 가장 오래 조용한 구간은 어디일까요?

**실행**. 터미널 A에 서버 로그를 띄워 두고, 터미널 B에서 돌립니다.

```bash
# 3편: bash practice/ch03/run.sh python practice/ch03/train.py
#      (= docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= workspace python practice/ch03/train.py)
docker compose run --rm workspace python practice/ch03/train.py
```

### 코드가 지나가는 길

**① 무엇이 달라졌나**: 3편 명령에서 두 가지가 빠졌습니다.

- `-e MLFLOW_TRACKING_URI=`가 없으므로 compose가 준 `http://mlflow:5000`([`docker-compose.yml:28`](../../docker-compose.yml#L28))이 그대로 남습니다.
- `--no-deps`가 없으므로 compose가 workspace의 의존 관계(290-292줄)를 따라 mlflow가 healthy인지 확인하고 시작합니다. 이미 떠 있으면 그냥 지나갑니다.

**MLflow가 저장소를 고르는 곳**: 트래킹 주소의 앞부분(scheme)으로 저장소 클래스를 고릅니다(`mlflow/tracking/_tracking_service/utils.py:250-254`). `http`, `https`면 REST 저장소, `sqlite`, `postgresql` 같은 DB 주소면 SQLAlchemy 저장소입니다. `mlflow.log_params` 같은 함수는 그대로이고, 그 아래의 저장소만 바뀝니다. 서버는 받은 요청을 **3편에서 SQLite에 쓰던 바로 그 SQLAlchemy 저장소 코드**로 PostgreSQL에 씁니다.

**요청 로그**: 이 PC에서 run 하나가 보낸 요청은 35개였습니다. 3편 1단계의 지점과 맞춰 보면 이렇습니다. "초"는 첫 요청부터 지난 시간입니다.

| 초 | 서버 로그의 요청 (응답) | 3편 지점 | 무엇을 하는가 |
|---|---|---|---|
| 0 | `GET experiments/get-by-name` (**404**) | ② | 이름으로 찾았는데 없다. 404는 오류가 아니라 "없음"이라는 답이다 |
| 0 | `POST experiments/create`, `GET experiments/get` | ② | 만들고 다시 읽는다(`fluent.py:144`) |
| 0 | `POST runs/create` | ③ | RUNNING인 run을 만든다. 자동 태그 7개가 이 요청 본문에 함께 간다 |
| 1 | `POST runs/log-batch` | ④ | `set_tags` |
| 1 | `POST runs/log-batch` ×2 | ⑤ | `log_params` 두 번(85줄, 86-88줄) |
| 1 | `POST runs/log-inputs` ×2 | ⑤ | 학습·검증 데이터 요약(100줄) |
| | (요청 없음, 약 1초) | ⑥ | 학습. MLflow와 상관없는 구간 |
| 2 | `POST runs/log-batch` ×2 | ⑦ | 지표 6개(129줄), 곡선 167점(121줄) |
| 2 | `GET runs/get` | ⑧-1 전 | run의 파라미터를 읽어 모델에 복사할 준비 |
| 3 | `POST logged-models` | ⑧-1 | 모델 행을 PENDING으로 만들고 `m-…` id를 받는다 |
| 3 | `POST runs/outputs` | ⑧-2 | run → 모델 연결 |
| 3 | `GET runs/get`, `GET metrics/get-history` ×7 | ⑧-3 | 지표마다 **이력 전체**를 받아서 step 0인 값만 고른다 |
| 4 | `POST runs/log-batch` | ⑧-3 | 고른 값을 모델 id를 붙여 다시 기록한다(`logged_model_metrics`) |
| | (요청 없음, **약 15초**) | ⑧-4 | 클라이언트의 임시 폴더에서 파일 7개를 만든다. `uv export`가 이때 돈다 |
| 9 | `GET /api/3.0/mlflow/server-info` | (기록과 무관) | 클라이언트의 사용 통계(telemetry) 기능이 서버 종류를 묻는다(`mlflow/telemetry/client.py:277, 491`) |
| 19 | `GET logged-models/m-…` | ⑧-5 | 모델의 산출물 위치(`mlflow-artifacts:/1/models/m-…/artifacts`)를 읽는다 |
| 21 | `PUT mlflow-artifacts/artifacts/1/models/m-…/artifacts/<파일>` ×7 | ⑧-5 | 파일 하나에 요청 하나. 파일 내용이 이 요청 본문으로 서버에 간다 |
| 21 | `PATCH logged-models/m-…` | ⑧-6 | READY로 바꾼다 |
| 21 | `GET runs/get`, `POST runs/update` | ⑨ | run 이름을 읽어 🏃 줄을 찍고, FINISHED로 바꾼다 |

⑧-5의 PUT이 어떻게 버킷까지 가는지 양쪽 코드로 보면 이렇습니다.

- 클라이언트: `mlflow/store/artifact/http_artifact_repo.py:101`의 `log_artifacts`가 폴더를 돌며 파일마다 80줄의 `log_artifact`를 부르고, 96줄에서 PUT을 보냅니다. 앞 파일이 끝나야 다음 파일을 보냅니다.
- 서버: `mlflow/server/handlers.py:4010`의 `_upload_artifact`가 요청 본문을 받아 버킷에 씁니다. 이때 쓰는 키가 위 표의 `AWS_*`입니다.

### 관찰

**터미널 B**: 3편과 같은 줄들이 나오고, 끝에 두 줄이 더 있습니다.

```text
🏃 View run manual-lr0.05-leaves31 at: http://mlflow:5000/#/experiments/1/runs/…
🧪 View experiment at: http://mlflow:5000/#/experiments/1
```

이 두 줄은 REST 저장소일 때만 찍힙니다(`mlflow/tracking/_tracking_service/client.py:755-756`의 `if not isinstance(self.store, RestStore): return`). 3편에 없던 이유입니다. 주소의 `mlflow`는 compose 네트워크 안에서만 통하는 이름이라 Windows 브라우저에서는 열리지 않습니다. `mlflow:5000`을 `127.0.0.1:5000`으로 바꿔서 엽니다.

JSON의 `mae`는 3편과 같습니다(lr 0.05에서 1.7903). `experiment_id`는 3편과 다릅니다.

**터미널 A**: 위 표의 요청이 실시간으로 흐릅니다. 두 곳을 봅니다.

- ⑧-3의 `metrics/get-history`: `valid_l1_by_tree`는 167점을 모두 받아서 step 0 하나만 씁니다. 곡선이 1만 점이었다면 1만 점을 받습니다.
- ⑧-4의 15초 공백: 로그가 멈춘 동안 터미널 B에는 `Exported 205 dependencies via uv`가 찍힙니다. 서버는 이 시간 동안 할 일이 없습니다.

**UI** http://127.0.0.1:5000: 3편과 같은 화면입니다. 주소가 달라서 브라우저 설정이 따로 저장되므로, 처음 열면 왼쪽 위에서 **Model training**을 다시 고르고 오른쪽 MLflow Assistant 패널을 닫습니다. run 화면의 Artifacts 탭에 보이는 파일 7개는 서버가 버킷에서 읽어 온 목록입니다. 4-2에서 버킷을 끄면 이 탭이 "Artifact loading"에서 멈추는 것으로 확인됩니다.

나머지 run도 쌓아 둡니다. 2-4와 4단계에서 씁니다.

```bash
for lr in 0.02 0.1 0.2; do
  docker compose run --rm workspace python practice/ch03/train.py --param learning_rate=$lr
done
```

### 원리

- **코드가 그대로인 이유는 저장소 추상화입니다.** 기록 함수와 저장소 사이에 경계가 있고, 주소가 그 경계 아래를 고릅니다. 같은 이유로 3편에서 본 표와 4편 PostgreSQL의 표가 똑같습니다(2-2).
- **함수 호출이 네트워크 왕복이 됐습니다.** 3편에서는 공짜에 가깝던 호출 하나하나가 이제 요청 하나입니다. `train.py:119-121`에서 곡선 167점을 `log_metric` 167번이 아니라 `log_batch` 한 번으로 보낸 이유가 여기서 실제로 드러납니다. 167번이었다면 요청도 167개였습니다.
- **큰 데이터는 PUT 7개뿐입니다.** 나머지 28개는 작은 JSON입니다. 서버가 병목이 된다면 메타데이터보다 파일 쪽입니다(5단계의 표).
- **실험 id는 저장소마다 따로 매깁니다.** 그래서 코드는 실험을 id가 아니라 이름으로 찾습니다(`set_experiment("etth1-ot-6h")`). 첫 요청이 `get-by-name`인 이유입니다.
- 사용 통계 요청은 MLflow 개발진의 서버로 사용 정보를 보내는 기능이 붙인 것입니다. 바깥으로 나가는 통신을 막아야 하는 사내망에서는 `MLFLOW_DISABLE_TELEMETRY=true`(또는 `DO_NOT_TRACK=true`)로 끕니다.

### 2-2. 두 저장소 열어 보기

> 목표: UI 뒤에 있는 DB의 표와 버킷의 객체를 직접 보고, 둘이 무엇으로 이어지는지 안다.

**예측**: 버킷 안의 경로는 3편의 `mlruns/` 안 경로와 같은 모양일까요? DB와 버킷을 잇는 것은 무엇일까요?

**실행**

```bash
docker compose run --rm workspace python practice/ch04/inspect_server.py
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d mlflow -c "\dt"'
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d mlflow -c "select experiment_id, name, artifact_location from experiments"'
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d mlflow -c "select model_id, status, artifact_location from logged_models"'
```

### 코드가 지나가는 길

[`inspect_server.py`](inspect_server.py)는 두 저장소를 MLflow 서버를 거치지 않고 직접 엽니다. DB 계정과 S3 키를 쓰므로 운영자 권한으로 보는 셈입니다. 학습 코드는 이 권한을 쓰지 않습니다.

1. [`main`, 235줄](inspect_server.py#L235) → [`print_connection`, 173줄](inspect_server.py#L173): 여기까지는 학습 코드와 같은 MLflow 클라이언트로 서버에 묻습니다. 클라이언트가 아는 것(서버 주소, 실험의 산출물 위치)만 보여 줍니다. 산출물 위치를 보고 방식을 말해 주는 것이 [`artifact_mode`, 66줄](inspect_server.py#L66)입니다.
2. [`connect_db`, 110줄](inspect_server.py#L110): psycopg2로 PostgreSQL에 바로 붙습니다. 118줄에서 읽기 전용 세션으로 바꿉니다.
3. [`table_counts`, 132줄](inspect_server.py#L132): 3편 `inspect_store.py`의 표 설명(`TABLE_NOTES`)을 그대로 가져와 같은 표를 셉니다. 표 이름을 SQL에 넣을 때 문자열을 잇지 않고 식별자로 감쌉니다(140줄).
4. [`model_rows`, 145줄](inspect_server.py#L145): 모델 행과 그 모델을 만든 run을 함께 읽습니다([`MODEL_ROWS_SQL`, 43-49줄](inspect_server.py#L43-L49)). DB의 상태는 숫자라서 [`status_name`, 61줄](inspect_server.py#L61)이 이름으로 바꿉니다(1 PENDING, 2 READY, 3 UPLOAD_FAILED).
5. [`s3_client`, 122줄](inspect_server.py#L122): boto3로 SeaweedFS에 바로 붙습니다.
6. [`bucket_and_prefix`, 82줄](inspect_server.py#L82): DB에 적힌 `mlflow-artifacts:/1/models/m-…/artifacts`를 (버킷 `mlflow`, 키 앞부분 `1/models/m-…/artifacts`)로 바꿉니다. 서버가 업로드할 때 하는 변환과 같은 일입니다.
7. [`print_models`, 192줄](inspect_server.py#L192) → [`verdict`, 97줄](inspect_server.py#L97): DB의 모델 상태와 버킷의 파일 수를 맞대어 판단합니다.
8. [`print_orphans`, 208줄](inspect_server.py#L208): 반대 방향입니다. 버킷에는 폴더가 있는데 DB에는 행이 없는 모델을 찾습니다.

### 관찰

**`inspect_server.py`** (이 PC의 격리 스택, run 하나와 등록 하나를 한 뒤)

```text
== 1. 클라이언트가 아는 것
트래킹 서버: http://mlflow:5000
실험: etth1-ot-6h (id 1)
산출물 위치: mlflow-artifacts:/1
  → 프록시. 올릴 때는 클라이언트가 서버에 HTTP로 보내고 서버가 버킷에 쓴다. 받을 때는 서버가 서명해 준 주소로 버킷에서 직접 받는다. 어느 쪽이든 클라이언트에 S3 키가 필요 없다

== 2. PostgreSQL: 3편의 mlflow.db와 같은 표가 DB 서버에 있다
runs                            1행  run. 상태(RUNNING/FINISHED/FAILED), 시작·끝 시각, 산출물 위치
params                          8행  파라미터. (run_uuid, key, value) — 값은 문자열
metrics                       173행  지표의 모든 기록. (run_uuid, key, value, step, timestamp)
latest_metrics                  7행  지표마다 마지막 값만. run 목록 화면이 빨리 뜨게 따로 둔다
tags                            9행  run 태그. mlflow.source.git.commit 같은 자동 태그 포함
logged_model_metrics            7행  모델에 연결된 지표
...
(그 밖의 표 48개 중 행이 있는 표 1개. 대부분 GenAI 기능용)

== 3. DB의 모델 행과 버킷의 파일 맞대기
m-5813a863…  run=manual-lr0.05-leaves31(FINISHED)  모델=READY
    s3://mlflow/1/models/m-5813a863…/artifacts/  파일 7개, 346,617바이트
    → 정상

버킷에만 있고 DB에는 없는 모델 폴더: 없음
```

params 8, metrics 173, latest_metrics 7, tags 9. 3편 6단계에서 SQLite로 센 숫자와 같습니다.

**psql**: `\dt`는 표 65개를 보여 줍니다. 위에서부터 `alembic_version`, `assessments`, `budget_policies`, `datasets`, … 순서입니다. 아래 두 쿼리의 결과는 이렇습니다.

```text
 experiment_id |    name     |  artifact_location
             0 | Default     | mlflow-artifacts:/0
             1 | etth1-ot-6h | mlflow-artifacts:/1

              model_id              | status |                     artifact_location
 m-5813a863ee144fa5ba424a01b6044965 |      2 | mlflow-artifacts:/1/models/m-5813a863ee144fa5ba424a01b6044965/artifacts
```

**SeaweedFS 파일 화면**: 브라우저로 http://127.0.0.1:8888/buckets/mlflow/ 을 엽니다. 위에 "SeaweedFS Filer"와 경로(`/ buckets mlflow`)가 있고, 아래에 실험 id 이름의 폴더가 있습니다. `<실험 id>` → `models` → `m-…` → `artifacts`로 들어가면 파일 7개가 종류, 크기와 함께 보입니다. `model.skops`는 `application/zip`, 약 332 kB이고 `requirements.txt`는 4.5 kB입니다(3편 6단계의 205줄짜리 목록).

**캡처 `mlops-artifacts.png`**: 이 `artifacts` 폴더 화면을 찍습니다. 위의 경로(`buckets / mlflow / <실험 id> / models / m-… / artifacts`)와 파일 7개가 함께 보여야 합니다. 이 한 장이 "DB에 적힌 문자열 `mlflow-artifacts:/<id>/models/m-…/artifacts`가 실제로 가리키는 곳"을 보여 줍니다. 노트 04의 `<!-- 실습 2 (지난 글 실습을 서버로 옮기기)` 주석 안 `캡처:` 항목에 적힌 그림입니다. 본문의 "산출물 경로 구조" 트리 옆에 넣습니다.

**캡처 `mlops-db-tables.png`**: `\dt` 결과를 찍습니다. 65개를 다 담을 필요는 없고 `runs`, `params`, `metrics`, `logged_models`, `model_versions`가 보이는 부분이면 됩니다. 3편의 SQLite와 같은 표라는 것을 보여 주는 그림입니다. 같은 `캡처:` 항목에 있고, 코드 블록으로 대신해도 됩니다.

### 원리

- **산출물 위치가 경로가 아니라 주소입니다.** 3편에서는 실험의 `artifact_location`이 `/workspace/mlruns/1`이라는 컨테이너 안 절대 경로였습니다. 여기서는 `mlflow-artifacts:/1`입니다. 이 주소를 실제 버킷(`s3://mlflow/1/…`)으로 바꾸는 것은 서버의 `MLFLOW_ARTIFACTS_DESTINATION`입니다.
- **DB와 버킷을 잇는 것은 `artifact_location` 문자열 하나뿐입니다.** 외래 키도 트랜잭션도 없습니다. 그래서 한쪽에만 기록이 남는 일이 생길 수 있고(4-2), `inspect_server.py`가 양쪽을 맞대어 보는 이유가 이것입니다.
- **표는 3편과 같습니다.** 같은 MLflow가 같은 구조로 만들기 때문입니다. SQLite 파일이 DB 서버로 옮겨 갔을 뿐입니다. 표 구조의 버전은 `alembic_version` 표에 있어서, MLflow를 올리면 서버가 이 값을 보고 표를 고칩니다.
- **버킷은 폴더처럼 보이지만 "키 → 내용"의 평평한 저장소입니다.** `1/models/m-…/artifacts/MLmodel`이 키 하나이고, 화면은 `/`를 기준으로 폴더처럼 묶어 보여 줄 뿐입니다. [`bucket_model_ids`, 162줄](inspect_server.py#L162)이 `Delimiter="/"`로 "폴더 목록"을 얻는 것도 같은 원리입니다.

### 2-3. 두 클라이언트가 동시에 기록하기

> 목표: 노트의 "다른 기기나 다른 컨테이너에서 같은 서버로 기록해 보기"를 확인한다.

**실행**

```bash
for lr in 0.03 0.3; do
  docker compose run --rm workspace python practice/ch03/train.py --param learning_rate=$lr &
done
wait
```

### 코드가 지나가는 길

컨테이너 두 개가 각자 프로세스로 1단계의 요청 35개씩을 보냅니다. 서버는 프로세스 2개(`MLFLOW_WORKERS`)로 요청을 받고, 각 요청은 PostgreSQL 연결을 하나 잡아 씁니다. 두 run의 행은 run id가 달라서 서로 다른 행이므로 부딪히지 않습니다.

### 관찰

- 터미널 A: 요청 줄 앞의 출발 주소(`172.…:포트`)가 두 가지로 섞여 나옵니다. 두 클라이언트의 요청이 번갈아 들어오는 것입니다.
- 두 run이 모두 FINISHED로 같은 실험에 모입니다(이 PC에서 33초). UI의 run 목록에 둘 다 보입니다.

### 원리

3편의 SQLite는 파일 하나를 통째로 잠가 가며 씁니다. 여러 사람이 네트워크 드라이브에 둔 `mlflow.db`에 동시에 쓰면 잠금이 불안정해집니다. 여기서는 동시 쓰기를 PostgreSQL이 행 단위로 처리하고, 클라이언트는 서버에 HTTP 요청만 보냅니다. 클라이언트끼리는 서로를 모릅니다.

### 2-4. 레지스트리와 모델 내려받기

> 목표: 등록, 별칭, 불러오기가 서버에서 어떤 요청이 되는지 보고, 모델 파일이 **어느 길로 내려오는지** 확인한다.

**예측**: `predict.py`가 모델 파일 7개를 받을 때, 그 파일 내용은 MLflow 서버를 지나갈까요? 서버 로그에 무엇이 찍힐까요?

**실행**. 3편의 실습 2 명령을 `run.sh` 없이 돌립니다. 터미널 A를 계속 봅니다.

```bash
R="docker compose run --rm workspace python practice/ch03/register.py"
$R register
$R alias 1
docker compose run --rm workspace python practice/ch03/predict.py
$R list
```

### 코드가 지나가는 길

**등록과 별칭**: 3편 8단계, 10단계와 같은 코드입니다. 요청으로는 이렇게 보입니다.

| 요청 | 코드 |
|---|---|
| `GET experiments/get-by-name`, `POST runs/search` | [`register.py:50-56`](../ch03/register.py#L50-L56) `search_runs` |
| `GET runs/get`, `POST logged-models/search` | [`register.py:73-80`](../ch03/register.py#L73-L80) run의 실험 id를 읽고, 그 run에서 나온 모델을 찾는다 |
| `POST registered-models/create`, `POST model-versions/create` | [`register.py:93`](../ch03/register.py#L93) `register_model` |
| `GET logged-models/m-…` ×2, `PATCH logged-models/m-…/tags` | 같은 줄. 모델에 `mlflow.modelVersions` 태그를 붙인다 |
| `GET registered-models/get` ×2, `POST registered-models/alias` | [`register.py:102, 108`](../ch03/register.py#L98-L109) 지금 별칭을 읽고 옮긴다 |

`register` 출력에 3편에는 없던 줄이 하나 섞입니다.

```text
INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: etth1-ot, version 1
```

`mlflow/store/model_registry/abstract_store.py:496`의 문구입니다. 3편의 SQLAlchemy 저장소는 이 함수를 "기다리지 않는다"로 덮어씁니다(`sqlalchemy_store.py:1599`). 버전을 만드는 순간 READY이기 때문입니다. REST 저장소는 덮어쓰지 않아서 기본 구현이 돌고, 문구를 찍은 뒤 상태를 확인합니다. 서버가 돌려준 상태가 이미 READY라서 실제로 기다리지는 않습니다. 로그에도 상태를 다시 묻는 요청이 없습니다.

**불러오기**: [`predict.py:37`](../ch03/predict.py#L37) → [`predict.py:50`](../ch03/predict.py#L50)

| 요청 | 하는 일 |
|---|---|
| `GET registered-models/alias?name=etth1-ot&alias=champion` | 37줄. 별칭 → 버전 |
| `GET registered-models/get`, `GET model-versions/get-download-uri` | 50줄. `models:/etth1-ot/1` → 그 버전의 파일 위치 |
| `GET mlflow-artifacts/artifacts?path=1/models/m-…/artifacts` ×2 | 폴더 안의 파일 목록 |
| `GET /api/3.0/mlflow/server-info` | 서버가 무엇을 지원하는지 묻는다 |
| `GET mlflow-artifacts/presigned/1/models/m-…/artifacts/<파일>` ×7 | 파일마다 **서명된 주소**를 받는다 |

여기서 로그가 끝납니다. 파일 7개의 내용을 받는 요청은 서버 로그에 **없습니다**. 클라이언트 코드가 그 이유입니다.

```python
# mlflow/store/artifact/mlflow_artifacts_repo.py:196-239 (줄임)
def _is_multipart_download_enabled(self):
    # 환경 변수로 정했으면 그대로 따른다
    if MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD.is_set():
        return MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD.get()
    # 아니면 서버에 묻는다(server-info)
    return self._fetch_server_capabilities().get(SERVER_INFO_MULTIPART_DOWNLOADS_ENABLED, False)


def _download_file(self, remote_file_path, local_path):
    if self._is_multipart_download_enabled():
        # 서버에 서명된 주소를 받아서
        presigned_response = self._get_presigned_download_url(remote_file_path)
        ...
        # 그 주소로 저장소에서 직접 받는다(나눠서 동시에)
        self._multipart_download(presigned_response=presigned_response, ...)
        return
    # 서버를 거쳐 받는다
    super()._download_file(remote_file_path, local_path)
```

서버 쪽에서는 1단계에서 본 `server-info`의 `multipart_downloads_enabled`가 이렇게 정해집니다.

```python
# mlflow/server/handlers.py:7494-7498 (서버 이미지, try 줄은 뺐다)
if _is_serving_proxied_artifacts():
    artifact_repo = _get_artifact_repo_mlflow_artifacts()  # s3://mlflow → S3 저장소
    multipart_uploads_enabled = isinstance(artifact_repo, MultipartUploadMixin)
    multipart_downloads_enabled = isinstance(artifact_repo, MultipartDownloadMixin)
```

설정이 아니라 "저장소 종류가 지원하는가"로 정해집니다. S3 저장소는 지원하므로 true가 되고, 클라이언트는 서명된 주소를 받아 갑니다. 서명은 서버의 `handlers.py:4361`(`_get_presigned_download_url`)이 서버의 키로 만들고, 유효 시간은 `MLFLOW_PRESIGNED_DOWNLOAD_URL_TTL_SECONDS`(기본 300초)입니다.

### 관찰

**터미널 B**: `predict.py`에 `Downloading artifacts` 진행 막대가 나오고, 2017Q3 MAE는 같은 모델이면 3편과 같습니다. UI의 Model registry에 `etth1-ot`이 생깁니다.

**터미널 A**: `presigned/…` 요청 7개 뒤에 파일을 받는 요청이 없는 것을 확인합니다.

**서명된 주소를 직접 받아 보기** (Windows의 WSL 터미널에서, 즉 compose 네트워크 밖에서). `<실험 id>`와 `<모델 id>`는 `inspect_server.py` 출력에서 가져옵니다.

```bash
P=<실험 id>/models/<모델 id>/artifacts/MLmodel
curl -s "http://127.0.0.1:5000/api/2.0/mlflow-artifacts/presigned/$P" | python3 -m json.tool
URL=$(curl -s "http://127.0.0.1:5000/api/2.0/mlflow-artifacts/presigned/$P" | python3 -c 'import sys, json; print(json.load(sys.stdin)["url"])')
curl -sS -o /dev/null "$URL"                                                   # ① 서명된 주소로 직접
curl -s "http://127.0.0.1:5000/api/2.0/mlflow-artifacts/artifacts/$P" | head -3   # ② 서버를 거쳐
```

이 PC에서 본 출력입니다(키 id와 서명은 가렸습니다).

```text
{
    "url": "http://seaweedfs:8333/mlflow/1/models/m-…/artifacts/MLmodel?response-content-disposition=…
            &X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=<키 id>/20261105/us-east-1/s3/aws4_request
            &X-Amz-Date=…&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=<서명>",
    "headers": {},
    "file_size": 1213
}
curl: (6) Could not resolve host: seaweedfs                       ← ①
artifact_path: mlflow-artifacts:/1/models/m-…/artifacts           ← ②
flavors:
  lightgbm:
```

같은 주소를 compose 네트워크 안(workspace 컨테이너)에서 열면 200으로 `MLmodel` 내용이 옵니다.

- 서명된 주소의 호스트는 `seaweedfs:8333`입니다. 서버가 아는 S3 주소(`MLFLOW_S3_ENDPOINT_URL`) 그대로입니다. compose 네트워크 밖에서는 이 이름을 찾지 못해 ①이 실패합니다.
- ②는 같은 파일을 서버를 거쳐 받는 길이고, 밖에서도 됩니다.
- 이 PC에서 compose 네트워크 밖의 컨테이너(`--network host`)로 `mlflow.pyfunc.load_model`을 돌렸더니, 몇 분 동안 재시도하다가 `Failed to download 1 chunk(s) … after all retries exhausted`로 실패했습니다. 같은 컨테이너에 `-e MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`를 주면 정상으로 불러왔습니다.

**서버를 거쳐 받게 바꿔 보기** (선택)

```bash
docker compose run --rm -e MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false workspace python practice/ch03/predict.py
```

이번에는 터미널 A에 `presigned`와 `server-info`가 없고, 대신 `GET mlflow-artifacts/artifacts/1/models/m-…/artifacts/<파일>` 7개가 찍힙니다. 파일 내용이 서버를 지나간 것입니다. MAE는 같습니다.

### 원리

- **노트의 "산출물 쓰기 방식" 표는 올리는 쪽만 맞습니다.** MLflow 3.17은 프록시 방식이어도, 서버의 저장소가 S3처럼 서명된 주소를 지원하면 **내려받기는 클라이언트가 저장소에서 직접** 합니다. 서버가 큰 파일을 중계하는 부담을 덜려는 설계입니다.
- **키는 여전히 필요 없지만, 저장소까지 네트워크가 닿아야 합니다.** 서명된 주소는 5분짜리 출입증입니다. 비밀 키는 들어 있지 않고 키 id와 서명만 있습니다. 대신 이 주소를 가진 사람은 누구든 5분 동안 그 파일을 받을 수 있으므로, 로그나 채팅에 붙여 넣지 않습니다.
- **주소 안의 호스트 이름이 문제가 됩니다.** 서버는 자기가 아는 이름(`seaweedfs`)으로 서명합니다. 노트북이나 다른 서버처럼 그 이름을 모르는 클라이언트는 모델을 불러오지 못합니다. 운영에서는 둘 중 하나를 고릅니다. 저장소를 클라이언트도 닿는 이름으로 서버에 알려 주거나, 클라이언트에 `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false`를 줘서 서버를 거치게 합니다. 앞의 것은 서버 부담이 적고, 뒤의 것은 저장소를 바깥에 열지 않아도 됩니다.
- 올리기에도 같은 장치가 있습니다. 코드로 보면 500MB 이상인 파일은 올릴 때도 서명된 주소로 저장소에 직접 나눠 올립니다(`mlflow_artifacts_repo.py:184-194`, `MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE`). 이 실습의 파일은 그보다 훨씬 작아서 확인하지 못했습니다.
- 레지스트리는 서버 DB에 있으므로, 서버에 닿는 클라이언트는 모두 같은 `etth1-ot@champion`을 봅니다. 3편의 로컬 레지스트리는 같은 폴더를 볼 수 있는 곳에서만 쓸 수 있었습니다.
- 3편 10단계에서 본 `registered_model_meta` 파일은 이번에는 버킷에 생기지 않습니다. 원격 저장소에서는 임시 폴더로 내려받고 그 사본에 쓰기 때문입니다.

### 노트에 쓸 것

- 코드에서 바뀐 부분: 없다. 접속 주소(`MLFLOW_TRACKING_URI`) 하나만 바뀌었다
- 산출물 경로 구조: `s3://mlflow/<실험 id>/models/<모델 id>/artifacts/{MLmodel, model.skops, …}`. 3편의 `mlruns/<실험 id>/models/…`와 모양이 같다
- 노트의 "산출물 쓰기 방식" 표에 "내려받기" 줄을 더할 것: 프록시 방식에서도 MLflow 3.17은 서명된 주소로 저장소에서 직접 받는다

| 항목 | 3편 로컬 | 4편 서버(프록시) |
|---|---|---|
| 클라이언트 설정 | `MLFLOW_TRACKING_URI` 비움 | `http://mlflow:5000` |
| MLflow 호출 하나 | 같은 프로세스 안의 함수 호출 | HTTP 요청 하나(run 하나에 약 35개) |
| 메타데이터 | `mlflow.db`(SQLite 파일) | PostgreSQL의 `mlflow` DB |
| 모델 파일 | `/workspace/mlruns/<id>/models/…` | `s3://mlflow/<id>/models/…` |
| 실험의 산출물 위치 값 | 절대 경로 `/workspace/mlruns/1` | 논리 주소 `mlflow-artifacts:/<id>` |
| 파일 올리기 | 폴더에 복사 | 클라이언트 → 서버(PUT) → 버킷 |
| 파일 받기 | 폴더를 그대로 연다 | 서버에 서명된 주소를 받고 → 버킷에서 직접 |
| 클라이언트에 필요한 것 | 그 폴더에 쓰기 권한 | 서버 주소. 그리고 받을 때 버킷에 닿는 네트워크 |
| 동시 기록 | 파일 잠금, 네트워크 드라이브에서 불안정 | DB가 처리 |
| 다른 컴퓨터에서 보기 | 같은 폴더가 보여야 함(경로까지 같아야 함) | 서버에 접속만 하면 됨(모델을 받으려면 버킷에도 닿아야 함) |

## 3. 컨테이너를 지워도 기록이 남는가

> 목표: 컨테이너와 상태(볼륨)를 구분한다.

**예측**: 컨테이너 셋을 지우고 다시 만들면 run과 모델 파일은 남을까요? run id는 그대로일까요?

**실행**. 볼륨은 지우지 않습니다.

```bash
docker compose rm -sf mlflow postgres seaweedfs  # 멈추고 컨테이너 삭제
docker volume ls | grep mlops                    # mlops_postgres-data, mlops_seaweedfs-data는 남아 있다
docker compose up -d --wait
docker compose ps                                # Up 몇 초: 새 컨테이너
docker compose run --rm workspace python practice/ch04/inspect_server.py
```

### 코드가 지나가는 길

- postgres는 [`docker-compose.yml:72`](../../docker-compose.yml#L72)에서 `postgres-data` 볼륨을 `/var/lib/postgresql`에 붙입니다. seaweedfs는 [97줄](../../docker-compose.yml#L97)에서 `seaweedfs-data`를 `/data`에 붙입니다. 이 이름들은 파일 끝의 `volumes:`(294줄~)에 선언되어 있고, compose가 프로젝트 이름을 앞에 붙여 `mlops_postgres-data`가 됩니다.
- `rm -sf`는 컨테이너만 지웁니다. `up`은 같은 이름의 볼륨이 있으면 새로 만들지 않고 그대로 붙입니다.
- mlflow 컨테이너에는 볼륨이 없습니다. 서버는 상태를 갖지 않고, 모든 상태는 DB와 버킷에 있습니다.

### 관찰

`ps`의 Up 시간이 몇 초로 바뀌었습니다(새 컨테이너). 그런데 `inspect_server.py`의 모델 행과 파일, UI의 run 목록과 run id, 레지스트리의 champion이 모두 그대로입니다.

### 원리

컨테이너는 이미지에서 언제든 다시 만들 수 있는 실행 환경이고, 상태(기록)는 볼륨에 있습니다. 반대로 `docker compose down -v`는 볼륨까지 지워서 모든 기록이 사라집니다. 운영에서 백업하는 대상은 컨테이너가 아니라 이 볼륨(또는 DB 덤프와 버킷)입니다. 그리고 2-2에서 봤듯 두 저장소는 문자열 하나로만 이어져 있으므로, 백업도 **같은 시점에** 둘 다 떠야 합니다.

## 4. 망가뜨리기

> 목표: 저장소 둘 중 하나가 죽으면 어디서, 얼마나 늦게, 무엇을 남기고 실패하는지 요청 로그로 본다.

**예측**: 각각 학습은 시작될까요? 몇 번째 요청에서 실패할까요? 실패하면 DB에는 무엇이 남을까요? `docker compose ps`의 mlflow는 무엇이라고 할까요?

**먼저 알아 둘 함정: `--no-deps`**. 아래 명령에는 모두 `--no-deps`가 붙어 있습니다. 1단계의 사슬을 떠올려 보세요. workspace는 mlflow에(290-292줄), mlflow는 postgres와 create-buckets에(145-149줄), create-buckets는 seaweedfs에(118-120줄) 묶여 있습니다. `--no-deps`를 빼면 `docker compose run`이 이 사슬을 따라 **꺼 둔 서비스를 먼저 다시 켜고** 학습을 돌립니다. 이 PC에서 빼고 돌려 보니 SeaweedFS가 다시 켜지고 create-buckets까지 다시 돈 뒤 학습이 정상으로 끝났습니다. 장애를 일부러 만들었는데 도구가 몰래 고쳐 버린 것입니다. 장애 실험에서는 "정말 그 상태에서 돌았는가"부터 확인해야 합니다.

`MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`은 재시도를 줄여 결과를 빨리 보려는 것입니다. 기본값은 아래 원리 4번에 있습니다.

### 4-1. PostgreSQL 끄기

**실행**

```bash
docker compose stop postgres
docker compose ps mlflow                         # mlflow 상태
curl -s -o /dev/null -w "health %{http_code}\n" http://127.0.0.1:5000/health
curl -s -o /dev/null -w "api %{http_code}\n" -X POST http://127.0.0.1:5000/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' -d '{"max_results":1}'
time docker compose run --rm --no-deps -e MLFLOW_HTTP_REQUEST_MAX_RETRIES=1 workspace python practice/ch03/train.py
```

이 상태에서 UI의 run 목록을 새로 고칩니다. 확인이 끝나면 다시 켭니다.

```bash
docker compose start postgres
```

#### 코드가 지나가는 길

학습 코드는 ②의 첫 요청에서 멈춥니다. 클라이언트 쪽 추적 메시지를 아래에서 위로 읽으면 이 순서입니다.

```text
train.py:184            mlflow.set_experiment(args.experiment)                       ②
fluent.py:208           client.get_experiment_by_name(experiment_name)
rest_store.py:1472      self._call_endpoint(GetExperimentByName, req_body)          REST 저장소
rest_utils.py:298       _get_http_response_with_retries(...)                        재시도 붙은 요청
urllib3 retry.py:555    MaxRetryError: … (Caused by ResponseError('too many 503 error responses'))
```

서버에서는 요청을 받은 SQLAlchemy 저장소가 DB 연결을 잡으려다 실패하고, 요청에 503(잠시 쓸 수 없음)으로 답합니다. 서버 로그입니다.

```text
"GET /api/2.0/mlflow/experiments/get-by-name?experiment_name=etth1-ot-6h HTTP/1.1" 503 Service Unavailable
"GET /api/2.0/mlflow/experiments/get-by-name?experiment_name=etth1-ot-6h HTTP/1.1" 503 Service Unavailable
ERROR mlflow.store.db.utils: SQLAlchemy database error. The following exception is caught.
(psycopg2.OperationalError) could not translate host name "postgres" to address: Name or service not known
```

요청이 두 번인 것은 처음 한 번 + 재시도 한 번이기 때문입니다. 오류가 "연결 거부"가 아니라 "이름을 찾을 수 없음"인 것도 볼 만합니다. 멈춘 컨테이너는 compose 네트워크의 이름 목록에서 빠지므로, `postgres`라는 이름 자체가 사라진 것입니다.

#### 관찰

- `ps`의 mlflow: `healthy`. `/health`: 200. API: 503.
- 학습: 첫 요청에서 실패합니다. 이 PC에서 컨테이너 시작을 포함해 약 6초였습니다.
- UI: run 목록 자리에 **`Experiment load error: (psycopg2.OperationalError) could not translate host name "postgres" to address`**가 뜹니다.
- 다시 켠 뒤: MLflow를 재시작하지 않아도 UI가 바로 돌아옵니다. 서버의 DB 연결 풀이 실패한 연결을 버리고, 다음 요청에서 새로 연결하기 때문입니다. 서버 로그 추적 메시지 첫머리의 `InvalidatePoolError`가 연결을 버린 흔적입니다.

**캡처 `mlops-db-down.png`**: UI의 이 오류 문구를 찍습니다. 같은 순간에 `docker compose ps`가 mlflow를 healthy라고 말하고 있다는 것이 이 그림의 요점입니다. 터미널을 옆에 띄워 함께 찍으면 더 좋습니다. 노트 04에는 아직 자리가 없습니다. `## 실습: compose로 기반 스택 띄우기` 끝에 "망가뜨려 보기" 절을 새로 만들어 넣습니다.

### 4-2. SeaweedFS 끄기

**실행**

```bash
docker compose stop seaweedfs
time docker compose run --rm --no-deps workspace python practice/ch03/train.py --param learning_rate=0.15
# 기다리는 동안 다른 터미널에서
docker compose ps mlflow
```

이번에는 기본 재시도 설정 그대로 한 번 돌려 보기를 권합니다. 몇 분 동안 아무 말 없이 멈춰 있는 것 자체가 볼 거리입니다. 빨리 보고 싶으면 `-e MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`을 붙입니다.

학습이 실패로 끝나면 **SeaweedFS를 켜기 전에** UI를 봅니다. 그다음 켜고 점검합니다.

```bash
docker compose start seaweedfs
docker compose run --rm workspace python practice/ch04/inspect_server.py
```

#### 코드가 지나가는 길

서버 로그를 2-1의 표와 나란히 놓고 봅니다. ⑧-4까지는 2-1과 **똑같이 모두 200**입니다. 메타데이터는 PostgreSQL로 가고, PostgreSQL은 살아 있기 때문입니다. 갈라지는 곳은 ⑧-5부터입니다(재시도 1회로 돌린 기록).

```text
00:25:05 "GET /api/2.0/mlflow/logged-models/m-d62a…" 200                                       ⑧-5 시작
00:25:12 "PUT /api/2.0/mlflow-artifacts/artifacts/1/models/m-d62a…/artifacts/model.skops" 500   첫 파일
00:25:19 "PUT /api/2.0/mlflow-artifacts/artifacts/1/models/m-d62a…/artifacts/model.skops" 500   재시도
00:25:19 "PATCH /api/2.0/mlflow/logged-models/m-d62a…" 200                                      ⑧-6: FAILED
00:25:19 "GET /api/2.0/mlflow/runs/get?…" 200                                                   ⑨: 🏃 줄
00:25:19 "POST /api/2.0/mlflow/runs/update" 200                                                 ⑨: FAILED
```

- 500의 원인은 서버 로그의 오류에 있습니다. `botocore.exceptions.EndpointConnectionError: Could not connect to the endpoint URL: "http://seaweedfs:8333/mlflow/1/models/m-d62a…/artifacts/model.skops"`. 서버의 `_upload_artifact`(`handlers.py:4010`)가 버킷에 쓰려다 실패한 것을 클라이언트에는 500으로 돌려줬습니다.
- 첫 파일(`model.skops`)에서 실패했으므로 나머지 6개는 시도조차 하지 않았습니다. 클라이언트의 추적 메시지는 `train.py:208` → `train.py:160` → `model.py:1308`(⑧-5) → `http_artifact_repo.py:113`(파일마다 반복) → `:96`(PUT)입니다.
- 예외가 `with _use_logged_model(...)` 블록을 빠져나가면서 3편 1단계에서 본 `except Exception:`이 실행됩니다. 그것이 `PATCH logged-models`(FAILED)입니다. DB에는 숫자 3(`UPLOAD_FAILED`)으로 들어갑니다.
- 예외는 계속 올라가 `with mlflow.start_run()`도 빠져나가고, ⑨의 `__exit__`가 FAILED로 끝냅니다. 그것이 `runs/update`입니다. 🏃 줄은 이때 찍혀서, 실패한 run인데도 터미널에 링크가 나온 뒤에 추적 메시지가 이어집니다.

#### 관찰

- 터미널: 학습은 끝까지 돌고, 모델 파일을 올리는 요청에서 실패합니다. 이 PC에서 재시도 1회로 약 26초, 기본 재시도로 352초 걸렸습니다.
- `ps`의 mlflow: 이번에도 healthy입니다.
- UI run 목록: `manual-lr0.15-leaves31`이 빨간 X(Failed)입니다.
- UI 실험의 **Models**: 이 run의 모델이 Status **Failed**입니다. 그런데 오른쪽 `mae` 칸에 **1.7578**이 있습니다. 지금까지 본 어떤 모델보다 좋은 값입니다. 모델을 누르면 About의 Status도 Failed입니다. Artifacts 탭은 버킷이 꺼져 있어서 "Artifact loading"에서 멈춥니다.
- `inspect_server.py`:

  ```text
  m-d62a…  run=manual-lr0.15-leaves31(FAILED)  모델=UPLOAD_FAILED
      s3://mlflow/1/models/m-d62a…/artifacts/  파일 0개, 0바이트
      → 반쪽 기록: run의 지표는 DB에 있고 모델 파일은 0개
  ```

**캡처 `mlops-upload-failed.png`**: Models 화면에서 Status Failed와 mae 1.7578이 한 줄에 보이게 찍습니다. "가장 좋아 보이는 모델에 파일이 없다"는 것이 이 그림의 요점입니다. 4-1과 같은 "망가뜨려 보기" 절에 넣습니다.

### 이 PC에서 본 결과

| | PostgreSQL 끔 | SeaweedFS 끔 |
|---|---|---|
| `docker compose ps`의 mlflow | healthy | healthy |
| `/health` | 200 | 200 |
| 실패한 요청 | 첫 요청(②의 `experiments/get-by-name`)이 503 | 24번째(⑧-5의 첫 `PUT model.skops`)가 500. 실험이 이미 있어서 2-1보다 2개 적다 |
| 걸린 시간 | 약 6초(재시도 1회) | 약 26초(재시도 1회), 352초(기본 재시도) |
| 남은 기록 | 없음 | run은 FAILED지만 파라미터 8개, 지표 7개는 DB에 있음. 모델 행은 UPLOAD_FAILED(UI에는 Failed), 버킷의 파일은 0개 |
| UI | `Experiment load error` | 평소처럼 뜨고, Failed 모델에 mae가 보인다 |
| 다시 켠 뒤 | 서버를 재시작하지 않아도 바로 회복 | 같음 |

### 원리

1. **healthy는 "살아 있다"이지 "일한다"가 아니다.** compose의 healthcheck는 `/health`만 부르고, 1단계에서 본 대로 이 주소는 `"OK", 200`만 돌려줍니다. 그래서 둘 중 무엇이 죽어도 mlflow는 healthy로 보입니다. 의존 대상까지 확인하는 검사로 바꾸면 고장을 잡을 수 있습니다. 대신 DB가 잠깐 흔들릴 때 MLflow까지 재시작되는 연쇄가 생길 수 있습니다. 쿠버네티스가 liveness(살아 있나)와 readiness(요청을 받아도 되나)를 나누는 이유입니다.
2. **실패 위치에 따라 비용이 다르다.** DB가 죽으면 첫 요청에서 실패해서 잃는 것이 없습니다. 저장소가 죽으면 2-1의 표에서 거의 마지막 요청까지 가서, 학습이 다 끝난 뒤에 실패합니다. 이 과제는 1초짜리 학습이라 괜찮지만, 몇 시간 걸리는 학습이라면 그 시간을 통째로 잃습니다. 학습 전에 저장소부터 확인하거나, 모델을 먼저 로컬에 저장하고 올리는 단계만 따로 다시 시도하는 방식(6편의 파이프라인)이 필요한 이유입니다.
3. **두 저장소는 한 번에 쓰이지 않는다.** 2-2에서 봤듯 DB와 버킷은 문자열 하나로만 이어져 있고, 둘을 함께 되돌리는 트랜잭션이 없습니다. 그래서 지표는 PostgreSQL에 들어갔는데 모델 파일은 없는 "반쪽 기록"이 생깁니다. 그리고 이 반쪽 기록의 mae가 가장 좋아 보였습니다. **지표만 보고 모델을 고르면 파일이 없는 모델을 고르게 됩니다.** 3편의 `register.py`가 `attributes.status = 'FINISHED'`인 run만 후보로 고르는 이유이고, `inspect_server.py`처럼 둘을 맞대어 보는 점검이 필요한 이유입니다. (lr 0.15의 mae가 좋았던 것 자체는 진짜입니다. 3편 3단계의 학습률 탐색이 0.05~0.2 사이를 성기게 봤기 때문입니다. 탐색은 8편에서 다룹니다.)
4. **조용한 재시도.** MLflow 클라이언트는 500, 503 같은 응답을 받으면 기본으로 7번까지 다시 시도하고, 간격을 2배씩 늘립니다(`mlflow/environment_variables.py:233-253`의 `MLFLOW_HTTP_REQUEST_MAX_RETRIES=7`, `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=2`, 간격 최대 120초). 소스의 주석에는 "7번이면 약 4분"이라고 적혀 있습니다. 잠깐의 장애는 견디지만, 긴 장애에서는 몇 분 동안 아무 말 없이 멈춰 있다가 실패합니다. 4-2의 352초가 그것입니다. 파이프라인에서는 이 값과 전체 제한 시간을 일부러 정해 둡니다.

## 5. (선택) 직접 접근 방식과 비교하기

> 목표: 프록시 방식을 고른 이유를 반대쪽을 직접 써 보고 설명한다.

노트의 "시간이 되면 직접 접근 방식도 띄워 비교한다"입니다. 같은 PostgreSQL을 쓰고, 파일은 대신 올려 주지 않는 서버를 5002번 포트에 하나 더 띄웁니다.

**예측**: 클라이언트에 S3 키를 주지 않으면 어디서 실패할까요? 직접 접근으로 만든 모델을 나중에 프록시 서버(5000)로 접속해서 불러오면 될까요?

**실행**

```bash
docker compose -f docker-compose.yml -f practice/ch04/direct-access.yml up -d --wait mlflow-direct

# ① 클라이언트에 S3 키 없이
bash practice/ch04/direct.sh python practice/ch03/train.py --experiment etth1-ot-6h-direct
# ② 키와 S3 주소를 주고
bash practice/ch04/direct.sh --with-keys python practice/ch03/train.py --experiment etth1-ot-6h-direct
docker compose run --rm workspace python practice/ch04/inspect_server.py --experiment etth1-ot-6h-direct

# ③ 프록시 서버(5000)로 접속해서, 직접 접근 실험의 READY 모델을 키 없이 불러 보기 (모델 id는 위 출력에서)
docker compose run --rm workspace python -c "import mlflow; mlflow.pyfunc.load_model('models:/<모델 id>')"

docker compose -f docker-compose.yml -f practice/ch04/direct-access.yml rm -sf mlflow-direct   # 끄기
```

mlflow-direct가 떠 있는 동안 다른 compose 명령에서 `Found orphan containers` 경고가 나옵니다. 기본 compose 파일에 없는 서비스라서 나오는 것이고, 끄면 사라집니다.

### 코드가 지나가는 길

**서버**: [`direct-access.yml:23-24`](direct-access.yml#L23-L24)

```yaml
- --no-serve-artifacts                                  # 파일을 대신 올리고 내려 주는 일을 끈다
- --default-artifact-root=s3://${MLFLOW_BUCKET:-mlflow}/direct   # 새 실험의 산출물 위치
```

이 서버로 만든 실험은 산출물 위치가 `mlflow-artifacts:/…`가 아니라 `s3://mlflow/direct/<실험 id>`로 DB에 저장됩니다.

**클라이언트**: [`direct.sh:30`](direct.sh#L30)이 서버 주소를 `http://mlflow-direct:5002`로 바꾸고 재시도를 1회로 줄입니다. ⑧-5에서 MLflow는 모델의 산출물 위치가 `s3://`로 시작하는 것을 보고, 서버에 PUT하는 대신 boto3로 버킷에 직접 씁니다. 그래서 boto3가 읽는 세 값이 클라이언트에 있어야 합니다. `--with-keys`일 때만 [`direct.sh:32-34`](direct.sh#L32-L34)가 workspace의 `S3_*` 값을 `AWS_*` 이름으로 바꿔 넘기고 `MLFLOW_S3_ENDPOINT_URL`을 줍니다. 1단계에서 본 대로, 이렇게 이름을 바꿔 주지 않으면 workspace의 키는 MLflow에게 보이지 않습니다.

### 관찰 (이 PC)

- ① 학습은 끝났고, 모델을 올리는 순간 `NoCredentialsError: Unable to locate credentials`가 났습니다. 이번에도 run은 FAILED, 모델 행은 UPLOAD_FAILED입니다. 4-2와 달리 이 오류는 서버의 응답(500)이 아니라 클라이언트 안의 boto3가 낸 것입니다. 키를 찾지 못해서 버킷에 요청을 보내 보지도 못했습니다.
- ② 성공했습니다. 실험의 산출물 위치가 `s3://mlflow/direct/<id>`이고, 파일은 클라이언트가 SeaweedFS에 직접 올렸습니다. SeaweedFS 파일 화면의 `buckets/mlflow/direct/`에서 보입니다.
- ③ 프록시 서버를 거쳤는데도 `NoCredentialsError`가 났습니다.

### 원리

산출물 위치는 실험을 만들 때 그 서버의 설정으로 정해져 DB에 저장됩니다. 그래서 나중에 어느 서버로 접속하든, 이 실험의 파일은 `s3://`로 직접 읽어야 합니다. 같은 이유로, 서버의 저장소 설정을 바꿔도 이미 있는 실험의 파일은 옮겨지지 않습니다.

| | 프록시 | 직접 접근 |
|---|---|---|
| 학습하는 쪽에 필요한 것 | 서버 주소(받을 때는 버킷에 닿는 네트워크도) | 서버 주소, S3 주소, S3 키 |
| 키를 가진 곳 | 서버 하나 | 학습하는 모든 곳 |
| 올릴 때 파일이 지나는 길 | 클라이언트 → 서버 → 저장소 (500MB 이상은 서명된 주소로 직접) | 클라이언트 → 저장소 |
| 받을 때 파일이 지나는 길 | 서명된 주소로 클라이언트 ← 저장소 (MLflow 3.17 기본) | 클라이언트 ← 저장소 |
| 약점 | 작은 파일 업로드가 모두 서버를 지난다. 서명된 주소의 호스트에 클라이언트가 닿아야 한다 | 키를 나눠 주고 바꾸는 일, 팀마다 권한을 나누는 일이 늘어난다 |

## 생각해 볼 질문

다음 확인 질문으로 이어집니다. 실습을 하면서 답을 생각해 두세요.

1. 4-2에서 학습이 4시간짜리였다면 무엇을 바꾸겠습니까? 학습 코드, 파이프라인, 저장소 중 어디를 고치는 것이 맞을까요?
2. mlflow의 healthcheck를 "DB와 버킷까지 확인"하도록 바꾸면 무엇을 얻고 무엇이 위험해집니까?
3. 팀 20명이 모델 하나에 5GB씩 올린다면 프록시와 직접 접근 중 무엇을 고르겠습니까? 2-4에서 본 서명된 주소는 이 선택을 어떻게 바꿉니까?
4. 장애 복구를 위해 PostgreSQL만 백업하고 버킷은 백업하지 않았습니다. 복구한 뒤 MLflow는 어떤 상태일까요? 반대로 버킷만 백업했다면요?
5. 사내 노트북에서 `etth1-ot@champion`을 불러와야 합니다. 서버(5000번)는 노트북에서 닿고, SeaweedFS는 닿지 않습니다. 2-4의 원리에 있는 두 방법 중 무엇을 고르겠습니까? 고른 방법에서 무엇을 더 확인해야 합니까?

## 처음부터 다시 하기

```bash
bash practice/ch04/reset.sh
```

4편 실험 둘(`etth1-ot-6h`, `etth1-ot-6h-direct`)과 등록 모델 `etth1-ot`만 DB와 버킷에서 지웁니다. 되돌릴 수 없습니다. 스택 테스트 기록 같은 다른 실험은 건드리지 않습니다.

[`reset.sh`](reset.sh)가 두 단계인 이유 자체도 배울 거리입니다. MLflow의 삭제는 두 단계입니다.

1. **삭제 표시**([`reset.sh:16`](reset.sh#L16) → `reset.py`): UI나 API(`delete_experiment`)로 지우면 `lifecycle_stage`만 `deleted`로 바뀝니다. DB 행과 버킷 파일은 그대로 남아서 되살릴 수 있습니다. 그런데 이름도 계속 차지합니다. 같은 이름으로 다시 기록하려 하면 `Cannot set a deleted experiment 'etth1-ot-6h' as the active experiment`가 납니다.
2. **완전 삭제**([`reset.sh:39-41`](reset.sh#L39-L41)): `mlflow gc`가 삭제 표시된 것을 DB와 버킷에서 실제로 지웁니다. gc는 HTTP API가 아니라 DB에 직접 붙으므로, DB 주소와 저장소 키를 가진 mlflow 서버 컨테이너 안에서 돌립니다(`docker compose exec mlflow`). 1단계에서 본 "비밀은 서버에만"의 결과입니다.

UI에서 지운 실험의 모델 파일이 버킷에 계속 쌓여 있는 것은 운영에서 흔한 저장 비용 문제입니다. 개인정보가 든 데이터라면 삭제 요청에도 대응하지 못한 셈이 됩니다. 그래서 운영에서는 gc를 주기적으로 돌립니다.

gc의 범위는 직관과 다릅니다. `--experiment-ids`만 주면 그 실험만 지울 것 같지만, `--run-ids`를 빼면 **DB 전체에서 삭제 표시된 모든 run**을, `--logged-model-ids`를 빼면 **삭제 표시된 모든 모델**을 함께 지웁니다(MLflow 3.17.2 소스로 확인). 실제로 이 실습을 만들면서 그렇게 돌렸다가, 스택 테스트가 남긴 삭제 표시된 run 10개를 지웠습니다. 그래서 `reset.py`는 지울 run id와 모델 id를 DB에서 직접 읽어 항상 넘깁니다(이미 삭제 표시된 것은 API 검색에 나오지 않습니다). 넘길 id가 없는데 다른 곳에 삭제 표시된 것이 있으면 멈춥니다. 되돌릴 수 없는 명령은 범위를 문서가 아니라 소스나 작은 실험으로 확인한 뒤에 돌립니다.

## 참고 문서

- [MLflow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/): 프록시와 직접 접근
- [백엔드 저장소](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [산출물 저장소](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
- [`mlflow gc`](https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-gc), [환경 변수 목록](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.environment_variables.html)
- [SeaweedFS S3 API](https://github.com/seaweedfs/seaweedfs/wiki/Amazon-S3-API), [Compose 시작 순서](https://docs.docker.com/compose/how-tos/startup-order/), [`docker compose run`의 `--no-deps`](https://docs.docker.com/reference/cli/docker/compose/run/)
