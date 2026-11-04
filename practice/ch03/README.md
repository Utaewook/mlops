# 3편 실습: 실험 추적과 모델 레지스트리

"MLOps 입문" 3편의 실습 0~2를 코드로 따라가는 자리입니다. 서버, DB, 오브젝트 스토리지 없이 MLflow를 **로컬에만** 씁니다.

코드는 모두 채워져 있습니다. 각 MLflow 호출 옆의 주석에 그 호출이 무엇을 만들고 어디에 쓰는지를 적었습니다. 아래 순서대로 파일을 읽고 명령을 돌리면서, 화면과 저장소에 실제로 무엇이 생기는지 확인하면 됩니다.

| 파일 | 노트 | 하는 일 |
|---|---|---|
| `run.sh` | - | workspace 컨테이너에서 명령을 돌린다. 트래킹 서버 주소를 비워 로컬에만 기록한다 |
| `train.py` | 실습 1 | 학습 한 번을 run 하나로 기록한다. `--autolog`를 주면 autolog로 기록한다 |
| `inspect_store.py` | 실습 1 | `mlflow.db`의 표와 `mlruns/` 폴더에 무엇이 들어갔는지 직접 열어 보여 준다 |
| `register.py` | 실습 2 | 모델을 레지스트리에 버전으로 등록하고, 별칭을 붙이고, 버전에서 run까지 거슬러 올라간다 |
| `predict.py` | 실습 2 | 별칭이 가리키는 모델을 불러와 학습에 쓰지 않은 분기(2017Q3)를 예측한다 |

## 전체 흐름

```
train.py ─ run 기록 ─┬─▶ mlflow.db        실험, run, 파라미터, 지표, 태그, 데이터 요약, 모델 정보, 레지스트리
                     └─▶ mlruns/1/models/m-…/artifacts/   모델 파일(model.skops, MLmodel, requirements.txt …)

register.py register ─▶ 등록 모델 etth1-ot 의 버전 N  ──가리킴──▶ models:/m-…
register.py alias N  ─▶ etth1-ot@champion ──가리킴──▶ 버전 N

predict.py ─ models:/etth1-ot@champion ─▶ 모델을 불러와 2017Q3 예측
```

## 준비

이미지에 git이 들어가도록 한 번 다시 빌드합니다(이 PC에서는 이미 했습니다). MLflow는 git 실행 파일이 있어야 run에 커밋 해시를 태그로 남깁니다.

```bash
docker compose --profile tools build workspace
```

데이터가 없으면 받아서 슬라이스로 나눕니다. `data/timeseries/`에 `2016Q3.parquet` 등이 있으면 건너뜁니다.

```bash
bash practice/ch03/run.sh python -m mlops prepare timeseries
```

기록은 저장소 루트의 `mlflow.db`(SQLite)와 `mlruns/`에 쌓입니다. 둘 다 `.gitignore`에 있어서 커밋되지 않습니다. 처음부터 다시 하려면 저장소 루트에서 `rm -rf mlflow.db mlruns`로 지웁니다.

## 실습 0. 데이터와 모델

노트의 후보(Bike Sharing, C-MAPSS) 대신 이 저장소의 Time Series 과제를 씁니다.

- 데이터: ETTh1(변압기 부하와 오일 온도, 1시간 간격, CC BY-ND 4.0)
- 문제: 6시간 뒤 오일 온도(OT) 예측. 2016Q3~2017Q1로 학습하고 2017Q2로 검증한다
- 모델: LightGBM 회귀. 온도가 아니라 **지금부터 6시간 동안의 변화량**을 예측하고, 지금 온도를 더해 6시간 뒤 온도를 낸다
- 기준선: "6시간 뒤에도 지금 온도 그대로". 지표의 `naive_mae`가 이것이고 `skill = 1 - mae / naive_mae`다

노트에 적을 버전은 다음 명령으로 확인합니다. 이 PC에서는 Python 3.12.16, MLflow 3.17.2, LightGBM 4.7.0입니다.

```bash
bash practice/ch03/run.sh python -c "import sys, mlflow, lightgbm; print(sys.version.split()[0], mlflow.__version__, lightgbm.__version__)"
```

## 실습 1. 실험 기록

### 1-1. `train.py` 읽기

맨 위 docstring에 MLflow가 기록을 담는 단위(실험, run, 파라미터, 지표, 태그, 입력, 모델)를 그림으로 적었습니다. 그다음 `main()`부터 읽고, 호출하는 함수로 내려갑니다.

1. `main()`: 실험 지정 → run 시작 → 태그 → 학습 전 기록 → 학습 → 학습 후 기록
2. `log_params_and_inputs()`: 파라미터와 데이터 요약을 학습 **전에** 남기는 이유
3. `log_results()`: 지표와 모델 기록. `log_model` 한 줄이 만드는 파일과 표를 주석에 풀어 두었다
4. `log_learning_curve()`: 같은 지표를 step별로 여러 번 남기면 곡선이 된다

### 1-2. run 쌓기

하이퍼파라미터를 바꿔 run을 여러 개 만듭니다. 바꿀 수 있는 값은 `--help`에 나옵니다. 한 번에 20~30초 걸립니다.

```bash
for lr in 0.02 0.05 0.1 0.2; do
  bash practice/ch03/run.sh python practice/ch03/train.py --param learning_rate=$lr
done
bash practice/ch03/run.sh python practice/ch03/train.py --param num_leaves=63
```

autolog run을 하나 만듭니다. 직접 기록한 `learning_rate=0.05` run과 설정이 같으므로 둘을 나란히 비교할 수 있습니다.

```bash
bash practice/ch03/run.sh python practice/ch03/train.py --autolog
```

### 1-3. UI에서 보기

다른 터미널에서 UI를 띄우고 Windows 브라우저로 http://127.0.0.1:5001 을 엽니다. 끌 때는 Ctrl+C입니다.

```bash
bash practice/ch03/run.sh ui
```

MLflow 3.17 UI는 GenAI 모드로 열립니다. 왼쪽 위의 **Model training**을 눌러 전환합니다. GenAI 모드에서는 같은 모델 목록이 "Agent versions"라는 이름으로 보입니다.

| 화면 | 들어가는 길 | 볼 것 |
|---|---|---|
| run 목록 | Experiments → `etth1-ot-6h` → Runs | run 이름, Dataset 칸. run을 둘 이상 체크해서 비교 화면으로 간다 (캡처 `mlops-runs.png`, `mlops-compare.png`) |
| run 화면 Overview | run 이름 클릭 | Source 칸의 파일 이름, 브랜치, git 커밋. Datasets, Tags, Metrics 표(Models 칸에 연결된 모델) |
| run 화면 Model metrics | 같은 화면의 탭 | `valid_l1_by_tree`(직접 기록) 또는 `valid_0-l1`(autolog)의 step별 곡선 |
| run 화면 Artifacts | 같은 화면의 탭 | 직접 기록한 run은 비어 있다. autolog run에는 feature importance 파일이 있다 |
| 모델 목록 | 실험의 Models | run마다 남긴 LoggedModel. 만든 파일(train.py)과 run이 같이 나온다 |

### 1-4. 저장소 들여다보기

UI가 보여 준 것이 실제로는 SQLite 표 몇 개와 파일 몇 개라는 것을 확인합니다.

```bash
bash practice/ch03/run.sh python practice/ch03/inspect_store.py
```

### 노트에 쓸 것

- run 하나에 무엇이 남는가: 파라미터, 지표(step 이력 포함), 태그(`mlflow.source.git.commit`, `mlflow.source.name`, `mlflow.user`), 데이터 요약, 모델
- autolog가 더 남기는 것: LightGBM의 모든 파라미터(기본값 포함 22개), 나무마다의 검증 지표, feature importance 그림, 데이터 요약, 모델
- autolog가 놓치는 것: 학습 함수(`lightgbm.train`) 바깥에서 계산한 값(`skill`, `naive_mae`, `rmse`)과 데이터 구간 같은 과제 정보. 콜백으로 넘긴 `early_stopping_rounds`도 빠진다
- 같은 설정도 이름이 다르다: 직접 기록은 sklearn 이름(`n_estimators`, `seed`), autolog는 LightGBM 내부 이름(`num_boost_round`, `random_state`). 두 방식을 섞으면 `params.n_estimators`로 검색할 때 autolog run이 빠진다
- autolog의 `valid_0-l1` 마지막 값은 직접 계산한 `mae`와 같다(이 PC에서 둘 다 1.79034). autolog는 검증 세트가 있으면 가장 좋았던 시점(67번째 나무)의 값을 곡선 끝(마지막 나무 다음 step)에 한 번 더 붙이기 때문이다. 직접 남긴 `valid_l1_by_tree`의 마지막 값은 멈춘 시점(167번째)의 값이라 조금 더 크다. 변화량의 MAE가 곧 온도의 MAE인 이유는 예측과 정답에 같은 "지금 온도"를 더하기 때문이다
- 로컬 저장 위치와 형식: 메타데이터는 `mlflow.db` 하나에 표로, 파일은 `mlruns/`에. 모델 파일이 run 폴더가 아니라 `mlruns/<실험>/models/<모델 id>/`에 있다
- MLflow 3.x의 변화: 모델이 run의 산출물에서 독립 객체(LoggedModel, `models:/m-…`)가 되었다. `log_model`의 `artifact_path` 인자가 `name`으로 바뀌었다

## 실습 2. 모델 레지스트리

### 2-1. `register.py`, `predict.py` 읽기

`register.py` 맨 위 docstring에 등록 모델, 버전, 별칭의 관계를 그림으로 적었습니다. 등록과 별칭 지정이 왜 따로인지가 핵심입니다.
`predict.py`에는 버전 번호도 run id도 없고 이름과 별칭만 있다는 점을 봅니다.

### 2-2. 등록하고, 별칭 붙이고, 옮기기

```bash
R="bash practice/ch03/run.sh python practice/ch03/register.py"

$R register                      # 검증 MAE가 가장 낮은 run의 모델 → 버전 1
$R register --run-id <다른 run id> # 다른 run의 모델 → 버전 2 (run id는 train.py 출력이나 UI에서)
$R alias 1                       # champion → 버전 1
bash practice/ch03/run.sh python practice/ch03/predict.py

$R alias 2                       # champion → 버전 2
bash practice/ch03/run.sh python practice/ch03/predict.py   # 코드는 그대로인데 모델과 MAE가 바뀐다

$R list                          # 버전마다 출처 run, mae, learning_rate, git 커밋
```

UI에서는 왼쪽 아래 **Model registry** → `etth1-ot`에서 버전 목록과 별칭을 봅니다(캡처 `mlops-registry.png`). 버전을 누르면 출처 run으로 이동할 수 있습니다.

### 노트에 쓸 것

- 버전 → run 추적: `register.py list`와 UI 모두 버전에서 run, 파라미터, 지표, git 커밋까지 따라갈 수 있다
- 로컬 레지스트리에 별도 DB가 필요했는가: 필요 없었다. 트래킹과 같은 `mlflow.db`에 `registered_models`, `model_versions`, `registered_model_aliases` 표로 들어간다. 모델 파일은 복사되지 않고 버전이 LoggedModel을 가리킨다
- 별칭을 옮기는 것만으로 쓰는 쪽의 모델이 바뀐다. 되돌리기도 별칭을 옮기면 된다

## 실행하면 보이는 경고

실습에는 영향이 없습니다.

- `UserWarning: Hint: Inferred schema contains integer column(s)`: 데이터 요약을 남길 때 정수 열(hour, dayofweek)이 결측값을 못 담는다는 안내입니다
- `Found torch version (2.15.0+cpu) contains a local version label`: 모델의 의존성 목록을 만들 때 `+cpu`를 떼고 적었다는 뜻입니다
- `Failed to resolve installed pip version`: 이미지의 가상환경을 uv로 만들어 pip이 없어서 나옵니다
- autolog run의 `` `artifact_path` is deprecated ``: MLflow의 autolog 코드가 스스로 옛 인자를 써서 나옵니다

run에 `mlflow.source.git.commit` 태그가 없으면 `bash practice/ch03/run.sh git status`를 돌려 봅니다. `detected dubious ownership`이 나오면 컨테이너 안에서 저장소 파일의 소유자가 실행 사용자와 다르게 보이는 환경(예: Docker Desktop)입니다. 이 PC(WSL2)에서는 태그가 남는 것을 확인했습니다.

## 4편으로 넘어가면

같은 코드를 트래킹 서버로 보내려면 4편 스택을 띄운 뒤 `run.sh` 대신 `docker compose run --rm workspace python practice/ch03/train.py`로 돌립니다. workspace의 기본 `MLFLOW_TRACKING_URI`가 `http://mlflow:5000`이라 코드는 한 줄도 바꾸지 않아도 됩니다. 이때 메타데이터는 PostgreSQL에, 모델 파일은 SeaweedFS에 들어갑니다. 로컬과 무엇이 달라지는지가 4편의 비교 거리입니다. 이어지는 실습은 [4편 실습 안내](../ch04/README.md)에 있습니다.

## 참고 문서

- [MLflow 3 소개](https://mlflow.org/docs/latest/ml/mlflow-3/): LoggedModel과 바뀐 모델 기록 방식
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [자동 기록(autolog)](https://mlflow.org/docs/latest/ml/tracking/autolog/)
- [백엔드 저장소](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/): 로컬 기본 저장 위치
- [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/), [mlflow.lightgbm API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.lightgbm.html)
