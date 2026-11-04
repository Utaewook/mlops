# 4편 실습: 기반 스택 (트래킹 서버 + PostgreSQL + SeaweedFS)

"MLOps 입문" 4편의 실습 1~2를 코드로 따라가는 자리입니다. 3편에서 로컬(`mlflow.db`, `mlruns/`)에 남기던 기록을 트래킹 서버로 보냅니다. 학습과 레지스트리 코드는 3편의 `practice/ch03/` 것을 **한 줄도 바꾸지 않고** 그대로 씁니다.

이 편에서 확인할 것은 세 가지입니다.

1. 무엇이 어디로 가는가: 메타데이터는 PostgreSQL의 표로, 모델 파일은 S3 버킷의 객체로
2. 무엇이 바뀌고 무엇이 그대로인가: 바뀌는 것은 접속 주소 하나
3. 어디가 고장 나면 무엇이 어떻게 깨지는가: 저장소 둘 중 하나를 꺼 보고, 키 없이 직접 써 본다

| 파일 | 실습 | 하는 일 |
|---|---|---|
| `inspect_server.py` | 2, 4, 5 | PostgreSQL의 표와 버킷의 객체를 직접 열어, DB의 모델 행과 버킷의 파일이 맞는지 맞대어 본다 |
| `direct-access.yml` | 5 | 산출물 "직접 접근" 방식의 MLflow 서버를 하나 더 띄우는 compose 파일 |
| `direct.sh` | 5 | 그 서버로 기록한다. `--with-keys`로 클라이언트에 S3 키를 줄지 고른다 |
| `reset.py`, `reset.sh` | 언제든 | 4편 실습 기록(실험 두 개와 등록 모델)을 DB와 버킷에서 완전히 지운다 |

## 전체 그림

```
                                      ┌── SQL ─────▶ PostgreSQL  DB "mlflow": 3편 mlflow.db와 같은 표
train.py ── HTTP ──▶ MLflow 서버 ─────┤
(workspace)          (mlflow:5000)    └── S3 API ──▶ SeaweedFS   버킷 "mlflow": <실험 id>/models/m-…/artifacts/…

아는 것: 서버 주소뿐         아는 것: DB 계정, S3 키, 버킷 이름
```

학습 코드는 서버 주소(`MLFLOW_TRACKING_URI=http://mlflow:5000`)만 압니다. DB 계정과 S3 키는 서버만 가지고, 모델 파일도 서버가 대신 버킷에 올립니다. 이것이 노트의 **프록시 방식**입니다.

## 준비

```bash
docker compose up -d --wait                      # postgres, seaweedfs, create-buckets, mlflow
docker compose ps                                # 상태가 healthy인지
docker compose run --rm workspace pytest -m base # 서비스끼리 실제로 연결되는지 (스택 테스트)
```

데이터는 3편에서 준비한 `data/timeseries/`를 그대로 씁니다. UI는 http://127.0.0.1:5000 입니다. 3편처럼 화면 위쪽에서 "Model training"을 고르면 Runs와 Models가 보입니다.

처음부터 다시 하려면 `bash practice/ch04/reset.sh`로 4편 기록을 지웁니다(맨 아래 [처음부터 다시 하기](#처음부터-다시-하기) 참고).

## 실습 1. 스택 읽기

`docker-compose.yml`의 "기본: 4편" 부분과 `infra/mlflow/Dockerfile`을 읽습니다. 노트에 쓸 "서비스마다 설정한 이유"는 다음과 같습니다.

| 서비스 | 이미지 | 설정 이유 |
|---|---|---|
| postgres | `postgres:18.6` | MLflow 메타데이터 DB. 볼륨을 붙여 컨테이너를 지워도 남게 한다. healthcheck를 TCP(`-h 127.0.0.1`)로 해야 첫 초기화가 끝난 뒤에 healthy가 된다 |
| seaweedfs | `chrislusf/seaweedfs:4.52` | S3 호환 객체 저장소. `mini` 모드로 마스터, 볼륨, 필러, S3 게이트웨이를 한 프로세스로 띄운다. S3 관리자 키는 `AWS_*` 환경 변수로 준다 |
| create-buckets | MLflow 이미지 재사용 | 버킷(`mlflow`, `data`)을 한 번 만들고 끝나는 일회성 컨테이너. 이미 있으면 건너뛴다 |
| mlflow | `ghcr.io/mlflow/mlflow:v3.17.2` + psycopg2, boto3 | 트래킹 서버. 공식 이미지에 DB 드라이버와 boto3가 없어서 추가 빌드한다. DB 주소, 버킷 주소, S3 키를 **이 서비스에만** 준다 |

시작 순서는 `depends_on`의 조건으로 묶습니다. 앞 단계가 준비되기 전에 MLflow가 뜨면 DB 연결이나 버킷 접근에서 실패하기 때문입니다.

```
postgres(healthy) ─────────────────────────────┐
seaweedfs(healthy) ─▶ create-buckets(성공 종료) ─┴─▶ mlflow(healthy) ─▶ workspace
```

구성하다 막혔던 부분(증상과 해결)은 저장소 루트 README의 "구성하며 확인한 것"에 있습니다. 예를 들어 MLflow 3.x 서버의 Host 헤더 검사, 작업 실행기의 메모리 사용, PostgreSQL 18의 데이터 경로 같은 것들입니다.

## 실습 2. 3편 실습을 서버로 옮기기

### 2-1. 같은 코드로 학습하기

3편과 다른 점은 명령의 `-e MLFLOW_TRACKING_URI=` 하나가 빠진 것뿐입니다. workspace 컨테이너의 기본값이 `http://mlflow:5000`이라서, 비우지 않으면 서버로 갑니다.

```bash
# 3편: bash practice/ch03/run.sh python practice/ch03/train.py
#      (= docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= workspace python practice/ch03/train.py)
docker compose run --rm workspace python practice/ch03/train.py
for lr in 0.02 0.1 0.2; do
  docker compose run --rm workspace python practice/ch03/train.py --param learning_rate=$lr
done
```

결과 숫자는 3편과 같습니다(lr 0.05의 검증 MAE 1.7903). 기록하는 곳만 달라졌기 때문입니다.

### 2-2. 두 저장소 열어 보기

```bash
docker compose run --rm workspace python practice/ch04/inspect_server.py
```

이 PC에서 본 출력의 일부입니다. id와 행 수는 환경마다 다릅니다.

```text
== 1. 클라이언트가 아는 것
트래킹 서버: http://mlflow:5000
실험: etth1-ot-6h (id 11)
산출물 위치: mlflow-artifacts:/11
  → 프록시. 클라이언트는 서버에 HTTP로 보내고 서버가 버킷에 쓴다. 클라이언트에 S3 키가 필요 없다

== 2. PostgreSQL: 3편의 mlflow.db와 같은 표가 DB 서버에 있다
runs              14행  run. 상태(RUNNING/FINISHED/FAILED), 시작·끝 시각, 산출물 위치
logged_models      9행  MLflow 3.x의 모델(m-...). 어느 run에서 나왔는지, 파일 위치
...
== 3. DB의 모델 행과 버킷의 파일 맞대기
m-eaca9e0c…  run=manual-lr0.05-leaves31(FINISHED)  모델=READY
    s3://mlflow/11/models/m-eaca9e0c…/artifacts/  파일 7개, 346,618바이트
    → 정상
```

볼 것:

- **산출물 위치가 경로가 아니라 주소다.** 3편에서는 실험의 `artifact_location`이 `/workspace/mlruns/1`이라는 컨테이너 안 절대 경로였습니다. 여기서는 `mlflow-artifacts:/11`이라는 논리 주소입니다. 이 주소를 실제 버킷(`s3://mlflow/11/…`)으로 바꾸는 것은 서버의 설정(`MLFLOW_ARTIFACTS_DESTINATION`)이 합니다. 그래서 버킷을 옮겨도 클라이언트는 바뀌지 않습니다.
- **클라이언트에 S3 키가 없다.** `docker compose run --rm workspace env | grep AWS`는 아무것도 출력하지 않습니다. 그래도 모델 파일 7개가 버킷에 올라갔습니다. 서버가 대신 올렸기 때문입니다. (workspace에 있는 `S3_ACCESS_KEY`는 이 점검 스크립트와 스택 테스트용입니다. MLflow는 이 이름을 읽지 않습니다.)
- **표는 3편과 같다.** 같은 MLflow가 같은 구조로 만들기 때문입니다. SQLite 파일이 DB 서버로 옮겨 갔을 뿐입니다.
- **실험 id가 3편과 다르다.** 로컬에서는 1이었는데 서버에서는 더 큰 수입니다. id는 저장소마다 따로 매기므로, 코드에서는 실험을 이름으로 찾습니다.

SeaweedFS의 파일 탐색 화면(http://127.0.0.1:8888/buckets/mlflow/)에서도 같은 경로가 보입니다. 노트의 캡처(`mlops-artifacts.png`)는 여기서 뜨면 됩니다. PostgreSQL 표 목록은 다음 명령으로 봅니다.

```bash
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d mlflow -c "\dt"'
```

### 2-3. 두 클라이언트가 동시에 기록하기

노트의 "다른 기기나 다른 컨테이너에서 같은 서버로 기록해 보기"입니다. 컨테이너 두 개를 동시에 돌립니다.

```bash
for lr in 0.03 0.3; do
  docker compose run --rm workspace python practice/ch03/train.py --param learning_rate=$lr &
done
wait
```

이 PC에서는 두 run이 모두 FINISHED로 한 실험에 모였습니다(33초). 3편의 SQLite는 파일 하나를 잠가 가며 쓰므로, 여러 사람이 네트워크 드라이브에 둔 `mlflow.db`에 동시에 쓰면 잠금이 불안정해집니다. 여기서는 동시 쓰기를 PostgreSQL이 맡습니다.

### 2-4. 레지스트리도 그대로

3편의 실습 2 명령을 `run.sh` 없이 돌립니다.

```bash
R="docker compose run --rm workspace python practice/ch03/register.py"
$R register
$R alias 1
docker compose run --rm workspace python practice/ch03/predict.py
$R list
```

이 PC에서는 2017Q3 MAE가 3편과 같은 1.6568이 나왔습니다. `predict.py`는 모델 파일을 서버를 거쳐 내려받습니다(`Downloading artifacts` 진행 막대). 이제 이 모델을 쓰는 다른 컴퓨터도 서버 주소만 알면 같은 `etth1-ot@champion`을 받습니다.

### 노트에 쓸 것

- 코드에서 바뀐 부분: 없다. 접속 주소(`MLFLOW_TRACKING_URI`) 하나만 바뀌었다
- 산출물 경로 구조: `s3://mlflow/<실험 id>/models/<모델 id>/artifacts/{MLmodel, model.skops, …}`. 3편의 `mlruns/<실험 id>/models/…`와 모양이 같다

| 항목 | 3편 로컬 | 4편 서버(프록시) |
|---|---|---|
| 클라이언트 설정 | `MLFLOW_TRACKING_URI` 비움 | `http://mlflow:5000` |
| 메타데이터 | `mlflow.db`(SQLite 파일) | PostgreSQL의 `mlflow` DB |
| 모델 파일 | `/workspace/mlruns/<id>/models/…` | `s3://mlflow/<id>/models/…` |
| 실험의 산출물 위치 값 | 절대 경로 `/workspace/mlruns/1` | 논리 주소 `mlflow-artifacts:/11` |
| 클라이언트에 필요한 권한 | 그 폴더에 쓰기 | 서버 주소만(키 없음) |
| 동시 기록 | 파일 잠금, 네트워크 드라이브에서 불안정 | DB가 처리 |
| 다른 컴퓨터에서 보기 | 같은 폴더가 보여야 함(경로까지 같아야 함) | 서버에 접속만 하면 됨 |

## 실습 3. 컨테이너를 지워도 기록이 남는가

컨테이너를 지우고 다시 만듭니다. 볼륨은 지우지 않습니다.

```bash
docker compose rm -sf mlflow postgres seaweedfs  # 멈추고 컨테이너 삭제
docker volume ls | grep mlops                    # mlops_postgres-data, mlops_seaweedfs-data는 남아 있다
docker compose up -d --wait
docker compose run --rm workspace python practice/ch04/inspect_server.py
```

run과 모델 파일이 그대로 있습니다. 컨테이너는 언제든 다시 만들 수 있는 실행 환경이고, 상태(기록)는 볼륨에 있기 때문입니다. 반대로 `docker compose down -v`는 볼륨까지 지워서 모든 기록이 사라집니다. 운영에서는 이 볼륨을 백업합니다.

## 실습 4. 망가뜨리기

저장소 둘 중 하나가 죽으면 어떻게 될까요? 돌리기 전에 먼저 예상해 보세요. 학습은 시작될까요? 어느 시점에 실패할까요? 실패하면 DB에는 무엇이 남을까요?

### 4-1. PostgreSQL 끄기

```bash
docker compose stop postgres
docker compose ps mlflow                         # mlflow 상태
curl -s -o /dev/null -w "health %{http_code}\n" http://127.0.0.1:5000/health
curl -s -o /dev/null -w "api %{http_code}\n" -X POST http://127.0.0.1:5000/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' -d '{"max_results":1}'
time docker compose run --rm -e MLFLOW_HTTP_REQUEST_MAX_RETRIES=1 workspace python practice/ch03/train.py
docker compose start postgres
```

`MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`은 재시도를 줄여 결과를 빨리 보려는 것입니다(기본값은 아래 4-2 참고).

### 4-2. SeaweedFS 끄기

```bash
docker compose stop seaweedfs
time docker compose run --rm workspace python practice/ch03/train.py --param learning_rate=0.15
# 기다리는 동안 다른 터미널에서
docker compose ps mlflow
docker compose logs --since 5m mlflow | grep -i error | tail -3
# 끝나면
docker compose start seaweedfs
docker compose run --rm workspace python practice/ch04/inspect_server.py
```

이번에는 기본 재시도 설정 그대로 한 번 돌려 보기를 권합니다. 몇 분 동안 아무 말 없이 멈춰 있는 것 자체가 볼 거리입니다. 빨리 보고 싶으면 `-e MLFLOW_HTTP_REQUEST_MAX_RETRIES=1`을 붙입니다.

### 이 PC에서 본 결과

| | PostgreSQL 끔 | SeaweedFS 끔 |
|---|---|---|
| `docker compose ps`의 mlflow | healthy | healthy |
| `/health` | 200 | 200 |
| 실패한 시점 | 학습 시작 전(`experiments/get-by-name`이 503) | 학습이 끝나고 모델 파일을 올릴 때(`model.skops` 업로드가 500) |
| 걸린 시간 | 5초(재시도 1회) | 352초(기본 재시도) |
| 남은 기록 | 없음 | run은 FAILED지만 파라미터 8개, 지표 7개는 DB에 있음. 모델 행은 UPLOAD_FAILED, 버킷의 파일은 0개 |
| 다시 켠 뒤 | 서버를 재시작하지 않아도 바로 회복 | 같음 |

여기서 배울 것:

1. **healthy는 "살아 있다"이지 "일한다"가 아니다.** compose의 healthcheck는 `/health`만 부르고, 이 주소는 DB도 버킷도 건드리지 않습니다. 그래서 둘 중 무엇이 죽어도 mlflow는 healthy로 보입니다. 의존 대상까지 확인하는 검사로 바꾸면 고장을 잡을 수 있습니다. 대신 DB가 잠깐 흔들릴 때 MLflow까지 재시작되는 연쇄가 생길 수 있습니다. 쿠버네티스가 liveness(살아 있나)와 readiness(요청을 받아도 되나)를 나누는 이유입니다.
2. **실패 위치에 따라 비용이 다르다.** DB가 죽으면 학습 전에 실패해서 잃는 것이 없습니다. 저장소가 죽으면 학습이 다 끝난 뒤에 실패합니다. 이 과제는 30초짜리라 괜찮지만, 몇 시간 걸리는 LLM 학습이라면 그 시간을 통째로 잃습니다. 학습 전에 저장소부터 확인하거나, 모델을 먼저 로컬에 저장하고 올리는 단계만 따로 다시 시도하는 방식(6편의 파이프라인)이 필요한 이유입니다.
3. **두 저장소는 한 번에 쓰이지 않는다.** 지표는 PostgreSQL에 들어갔는데 모델 파일은 없는 "반쪽 기록"이 생깁니다. 데이터베이스의 트랜잭션처럼 둘을 함께 되돌려 주지 않습니다. 그래서 `inspect_server.py`처럼 둘을 맞대어 보는 점검이 필요합니다. 3편의 `register.py`가 `attributes.status = 'FINISHED'`인 run만 후보로 고르는 것도 이 때문입니다. 반쪽 기록이 지표만 좋아 보여서 등록되는 일을 막습니다.
4. **조용한 재시도.** MLflow 클라이언트는 500, 503 같은 응답을 받으면 기본으로 7번까지 다시 시도하고, 간격을 2배씩 늘립니다(`MLFLOW_HTTP_REQUEST_MAX_RETRIES=7`, `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=2`, 간격 최대 120초). 잠깐의 장애는 견디지만, 긴 장애에서는 몇 분 동안 멈춰 있다가 실패합니다. 파이프라인에서는 이 값과 전체 제한 시간을 일부러 정해 둡니다.

## 실습 5. (선택) 직접 접근 방식과 비교하기

노트의 "시간이 되면 직접 접근 방식도 띄워 비교한다"입니다. 같은 PostgreSQL을 쓰고, 파일은 대신 올려 주지 않는 서버를 5002번 포트에 하나 더 띄웁니다. 설정 차이는 `direct-access.yml`의 주석에 있습니다.

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

이 PC에서 본 결과:

- ① 학습은 끝났고, 모델을 올리는 순간 `NoCredentialsError: Unable to locate credentials`가 났습니다. 이번에도 run은 FAILED, 모델 행은 UPLOAD_FAILED입니다.
- ② 성공했습니다. 실험의 산출물 위치가 `s3://mlflow/direct/12`이고, 파일은 클라이언트가 SeaweedFS에 직접 올렸습니다.
- ③ 프록시 서버를 거쳤는데도 `NoCredentialsError`가 났습니다. 산출물 위치는 실험을 만들 때 그 서버의 설정으로 정해져 DB에 저장됩니다. 그래서 나중에 어느 서버로 접속하든, 이 실험의 파일은 `s3://`로 직접 읽어야 합니다. 같은 이유로, 서버의 저장소 설정을 바꿔도 이미 있는 실험의 파일은 옮겨지지 않습니다.

| | 프록시 | 직접 접근 |
|---|---|---|
| 학습하는 쪽에 필요한 것 | 서버 주소 | 서버 주소, S3 주소, S3 키 |
| 키를 가진 곳 | 서버 하나 | 학습하는 모든 곳 |
| 파일이 지나는 길 | 클라이언트 → 서버 → 저장소 | 클라이언트 → 저장소 |
| 약점 | 큰 파일(LLM 가중치 등)이 모두 서버를 지나서 서버가 병목이 된다 | 키를 나눠 주고 바꾸는 일, 팀마다 권한을 나누는 일이 늘어난다 |

참고로 MLflow에는 프록시 방식에서도 큰 파일만 S3에 직접 나눠 올리게 하는 설정(`MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD`)이 있습니다. 이 저장소에서는 시험하지 않았습니다.

## 생각해 볼 질문

다음 확인 질문으로 이어집니다. 실습을 하면서 답을 생각해 두세요.

1. 4-2에서 학습이 4시간짜리였다면 무엇을 바꾸겠습니까? 학습 코드, 파이프라인, 저장소 중 어디를 고치는 것이 맞을까요?
2. mlflow의 healthcheck를 "DB와 버킷까지 확인"하도록 바꾸면 무엇을 얻고 무엇이 위험해집니까?
3. 팀 20명이 모델 하나에 5GB씩 올린다면 프록시와 직접 접근 중 무엇을 고르겠습니까? 무엇을 근거로 고르겠습니까?
4. 장애 복구를 위해 PostgreSQL만 백업하고 버킷은 백업하지 않았습니다. 복구한 뒤 MLflow는 어떤 상태일까요? 반대로 버킷만 백업했다면요?

## 처음부터 다시 하기

```bash
bash practice/ch04/reset.sh
```

4편 실험 둘(`etth1-ot-6h`, `etth1-ot-6h-direct`)과 등록 모델 `etth1-ot`만 DB와 버킷에서 지웁니다. 되돌릴 수 없습니다. 스택 테스트 기록 같은 다른 실험은 건드리지 않습니다.

이 스크립트가 두 단계인 이유 자체도 배울 거리입니다. MLflow의 삭제는 두 단계입니다.

1. **삭제 표시:** UI나 API(`delete_experiment`)로 지우면 `lifecycle_stage`만 `deleted`로 바뀝니다. DB 행과 버킷 파일은 그대로 남아서 되살릴 수 있습니다. 그런데 이름도 계속 차지합니다. 같은 이름으로 다시 기록하려 하면 `Cannot set a deleted experiment 'etth1-ot-6h' as the active experiment`가 납니다.
2. **완전 삭제:** `mlflow gc`가 삭제 표시된 것을 DB와 버킷에서 실제로 지웁니다. gc는 HTTP API가 아니라 DB에 직접 붙으므로, DB 주소와 저장소 키를 가진 서버 컨테이너에서 돌립니다.

UI에서 지운 실험의 모델 파일이 버킷에 계속 쌓여 있는 것은 운영에서 흔한 저장 비용 문제입니다. 개인정보가 든 데이터라면 삭제 요청에도 대응하지 못한 셈이 됩니다. 그래서 운영에서는 gc를 주기적으로 돌립니다.

gc의 범위는 직관과 다릅니다. `--experiment-ids`만 주면 그 실험만 지울 것 같지만, `--run-ids`를 빼면 **DB 전체에서 삭제 표시된 모든 run**을, `--logged-model-ids`를 빼면 **삭제 표시된 모든 모델**을 함께 지웁니다(MLflow 3.17.2 소스로 확인). 실제로 이 실습을 만들면서 그렇게 돌렸다가, 스택 테스트가 남긴 삭제 표시된 run 10개를 지웠습니다. 그래서 `reset.py`는 지울 run id와 모델 id를 DB에서 직접 읽어 항상 넘깁니다(이미 삭제 표시된 것은 API 검색에 나오지 않습니다). 넘길 id가 없는데 다른 곳에 삭제 표시된 것이 있으면 멈춥니다. 되돌릴 수 없는 명령은 범위를 문서가 아니라 소스나 작은 실험으로 확인한 뒤에 돌립니다.

## 참고 문서

- [MLflow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/): 프록시와 직접 접근
- [백엔드 저장소](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [산출물 저장소](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
- [`mlflow gc`](https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-gc), [환경 변수 목록](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.environment_variables.html)
- [SeaweedFS S3 API](https://github.com/seaweedfs/seaweedfs/wiki/Amazon-S3-API), [Compose 시작 순서](https://docs.docker.com/compose/how-tos/startup-order/)
