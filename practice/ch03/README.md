# 3편 실습: 실험 추적과 모델 레지스트리

"MLOps 입문" 3편의 실습서입니다. 서버, DB, 오브젝트 스토리지 없이 MLflow를 **로컬에만** 씁니다.

이 실습서는 명령을 따라 치게 하는 안내가 아닙니다. **명령 하나가 어떤 코드를 지나 MLflow 안에 무엇을 남기고, 그것이 화면 어디에 나타나는지**를 따라갑니다. 단계마다 순서가 같습니다.

1. **예측**: 돌리기 전에 결과를 짐작해서 적습니다. 틀린 곳이 배울 곳입니다.
2. **실행**: 명령을 돌립니다.
3. **코드가 지나가는 길**: 그 명령이 거치는 파일과 줄을 차례로 따라갑니다. 줄마다 MLflow의 어느 표, 어느 폴더에 무엇이 생기는지 적었습니다.
4. **관찰**: 그렇게 생긴 것이 터미널과 UI의 어디에 보이는지 확인합니다. 캡처는 이 안에 있습니다. 어느 화면을 찍는지, 그 화면이 무엇을 증명하는지, 노트의 어디에 넣는지까지 적었습니다.
5. **원리**: 왜 그렇게 되는지 정리합니다.

1단계에서는 `train.py` 한 번이 지나가는 길을 ①~⑩으로 끝까지 따라갑니다. 뒤의 단계는 "③에서 붙은 태그", "⑧-3에서 복사한 지표"처럼 이 번호로 되돌아옵니다. 그래서 1단계를 건너뛰지 않습니다.

코드 위치는 `파일:줄`로 적었습니다. GitHub에서 링크를 누르면 그 줄로 갑니다. 줄 번호는 이 커밋 기준입니다. MLflow 자체의 코드는 `mlflow/…:줄`로 적었습니다. 컨테이너 안의 `/opt/venv/lib/python3.12/site-packages/` 아래에 있고, 이렇게 엽니다.

```bash
# 예: run이 끝날 때 상태(FINISHED/FAILED)를 정하는 코드. 7단계에서 이 코드를 부숴 봅니다
bash practice/ch03/run.sh bash -c 'cd /opt/venv/lib/python3.12/site-packages && sed -n 383,393p mlflow/tracking/fluent.py'
```

"이 PC에서 본 값"은 2026-11-04~05에 MLflow 3.17.2로 확인한 값입니다. run id, 모델 id, 시각은 매번 다르게 나옵니다. 지표는 데이터와 시드가 같으므로 같게 나옵니다.

## 준비

이미지에 git이 들어가도록 한 번 다시 빌드합니다(이 PC에서는 이미 했습니다). MLflow는 git 실행 파일이 있어야 run에 커밋 해시를 남깁니다.

```bash
docker compose --profile tools build workspace
```

데이터가 없으면 받아서 분기별로 나눕니다. `data/timeseries/`에 `2016Q3.parquet` 등이 이미 있으면 건너뜁니다.

```bash
bash practice/ch03/run.sh python -m mlops prepare timeseries
```

기록은 저장소 루트의 `mlflow.db`(SQLite)와 `mlruns/`에 쌓입니다. 둘 다 `.gitignore`에 있어서 커밋되지 않습니다. 이 실습서의 값과 맞춰 보려면 빈 상태에서 시작합니다. 이전 기록은 지우지 말고 옮겨 둡니다.

```bash
mv mlflow.db mlflow.db.bak && mv mlruns mlruns.bak   # 되돌릴 때는 반대로
```

터미널은 두 개를 씁니다. 터미널 A는 UI용(2단계부터), 터미널 B는 명령용입니다.

캡처는 Windows에서 `Win + Shift + S`로 뜹니다. 노트 저장소의 `assets/images/MLOps/`에 단계에 적힌 이름으로 저장하고, 노트의 `<!-- 캡처 자리: ![[이름]] -->` 주석을 `![[이름]]`로 바꿉니다. UI 오른쪽의 MLflow Assistant 패널은 X로 닫고 찍어야 표가 넓게 나옵니다.

데이터와 모델은 노트의 실습 0과 같습니다. ETTh1로 6시간 뒤 오일 온도를 예측합니다. 2016Q3~2017Q1로 학습하고 2017Q2로 검증합니다. 모델은 온도가 아니라 **6시간 동안의 변화량**을 예측하고, 기준선은 "6시간 뒤에도 지금 온도 그대로"(`naive_mae`)입니다. 버전은 다음 명령으로 확인합니다(이 PC: Python 3.12.16, MLflow 3.17.2, LightGBM 4.7.0).

```bash
bash practice/ch03/run.sh python -c "import sys, mlflow, lightgbm; print(sys.version.split()[0], mlflow.__version__, lightgbm.__version__)"
```

---

# 실습 1. 실험 기록

## 1. 첫 run 하나: 명령 한 줄이 지나가는 길

> 목표: `train.py` 한 번이 MLflow 안에 무엇을 어떤 순서로 남기는지 끝까지 따라간다. 이후 모든 단계가 여기서 만든 기록을 본다.

**예측**: 이 명령 하나로 저장소 루트에 무엇이 생길까요? 파라미터, 지표, 모델 중 무엇이 가장 먼저 기록될까요? 모델 파일은 어느 폴더에 들어갈까요?

**실행** (터미널 B). 처음 한 번은 DB 표를 만드느라 2분쯤 걸립니다.

```bash
bash practice/ch03/run.sh python practice/ch03/train.py
```

### 코드가 지나가는 길

**① 컨테이너를 띄우고 기록할 곳을 정한다**: [`run.sh:28`](run.sh#L28)

```bash
exec docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= workspace "$@"
```

- workspace 컨테이너는 compose에서 `MLFLOW_TRACKING_URI=http://mlflow:5000`(4편의 서버 주소)을 받습니다([`docker-compose.yml:28`](../../docker-compose.yml#L28)). `-e MLFLOW_TRACKING_URI=`가 이 값을 빈 값으로 덮습니다.
- 주소가 비면 MLflow 3.x는 기본값 `sqlite:///mlflow.db`를 씁니다(`mlflow/store/tracking/__init__.py:13`). 상대 경로라서 작업 폴더 `/workspace`의 `mlflow.db`가 됩니다.
- `/workspace`는 저장소 폴더를 그대로 붙인 것입니다([`docker-compose.yml:288`](../../docker-compose.yml#L288)). 그래서 `--rm`으로 컨테이너가 지워져도 `mlflow.db`와 `mlruns/`는 저장소 루트에 남습니다.
- workspace는 mlflow 서버가 healthy여야 뜨도록 묶여 있습니다([`docker-compose.yml:290-292`](../../docker-compose.yml#L290-L292)). `--no-deps`가 이 묶음을 끊습니다. 이 옵션이 없으면 3편인데도 4편의 PostgreSQL, SeaweedFS, MLflow 서버가 함께 켜집니다.

**② 실험을 고른다**: [`train.py:184`](train.py#L184)

```python
mlflow.set_experiment(args.experiment)  # "etth1-ot-6h"
```

MLflow 안에서(`mlflow/tracking/fluent.py:144`) 이름으로 찾아보고(`get_experiment_by_name`), 없으면 만들고(`create_experiment`), 만든 것을 다시 읽습니다(`get_experiment`). 4편에서는 이 세 번이 HTTP 요청 세 개로 보입니다.

- 남는 것: `experiments` 표 1행. id 1, 이름, 산출물 위치(`artifact_location`) `/workspace/mlruns/1`
- 터미널: `Creating initial MLflow database tables...`(mlflow.db를 처음 만들 때만), `Experiment with name 'etth1-ot-6h' does not exist. Creating a new experiment.`

**③ run을 연다**: [`train.py:198`](train.py#L198)

```python
with mlflow.start_run(run_name=run_name(params, args.autolog)) as run:
```

`runs` 표에 1행이 생깁니다. 상태는 RUNNING이고 시작 시각이 들어갑니다. 이 순간 MLflow가 스스로 태그 7개를 붙입니다(`tags` 표).

| 자동 태그 | 이 PC 값 | 어디서 오나 |
|---|---|---|
| `mlflow.runName` | `manual-lr0.05-leaves31` | `run_name` 인자([`train.py:70-73`](train.py#L70-L73)) |
| `mlflow.user` | 호스트 사용자 이름 | workspace에 넘긴 `USER` 환경 변수([`docker-compose.yml:279`](../../docker-compose.yml#L279)) |
| `mlflow.source.name` | `practice/ch03/train.py` | 실행한 파일 |
| `mlflow.source.type` | `LOCAL` | |
| `mlflow.source.git.commit` | 40자리 해시 | `mlflow/utils/git_utils.py:71`의 `repo.head.commit.hexsha` |
| `mlflow.source.git.branch` | 지금 브랜치 | |
| `mlflow.source.git.repoURL` | 원격 저장소 주소 | |

**④ 직접 태그를 붙인다**: [`train.py:200`](train.py#L200)

```python
mlflow.set_tags({"task": TASK.name, "logging": "autolog" if args.autolog else "manual"})
```

`tags` 표에 `task=timeseries`, `logging=manual` 2행이 더 들어갑니다. 8단계의 `register.py`는 `tags.logging = 'manual'` 조건으로 등록할 run을 고르므로, 이 태그는 메모가 아니라 검색 조건입니다.

**⑤ 학습 전에 정해지는 것을 남긴다**: [`train.py:202`](train.py#L202) → `log_params_and_inputs` [`train.py:85-100`](train.py#L85-L100)

```python
mlflow.log_params(params)  # 85: 하이퍼파라미터 6개
mlflow.log_params({"train_slices": ..., "valid_slices": ...})  # 86-88: 학습·검증 구간
dataset = mlflow.data.from_pandas(df, name=..., targets="target")  # 99: 데이터 요약 만들기
mlflow.log_input(dataset, context=context)  # 100: training, validation
```

- `params` 표 8행. 값은 모두 **문자열**로 들어갑니다(0.05 → `"0.05"`). 3단계 검색에서 이것 때문에 걸립니다.
- 99줄의 `from_pandas`는 열 구조를 추론하면서 `UserWarning: Hint: Inferred schema contains integer column(s)`를 냅니다. 학습용과 검증용에서 **두 번** 나옵니다. 두 번째 경고가 나오면 ⑤가 끝나고 학습이 시작됐다는 뜻입니다. 7단계에서 이 신호를 씁니다.
- `datasets` 표 2행(이름, 지문, 열 구조, 행 수), `inputs` 표 2행(데이터 → run 연결), `input_tags` 표 2행(용도 training, validation). 데이터 자체는 올리지 않습니다.

**⑥ 학습한다**: [`train.py:204`](train.py#L204) → [`train.py:171`](train.py#L171) → [`mlops/tasks/timeseries.py:153-176`](../../mlops/tasks/timeseries.py#L153-L176)

MLflow와 상관없는 부분입니다. `LGBMRegressor.fit`이 나무를 최대 2000그루까지 하나씩 더합니다. 검증 MAE가 100그루 동안 나아지지 않으면 멈춥니다(조기 종료). 이 run은 67번째 나무가 가장 좋았고, 167그루에서 멈췄습니다.

**⑦ 지표를 남긴다**: [`train.py:208`](train.py#L208) → `log_results`

```python
mlflow.log_metrics({**metrics, "best_iteration": model.best_iteration_})  # 129
MlflowClient().log_batch(run_id, metrics=curve)  # 121 (130에서 부름)
```

- 129줄: 지표 6개(`mae`, `rmse`, `naive_mae`, `skill`, `rows`, `best_iteration`)가 step 0으로 들어갑니다. `metrics` 표 6행, `latest_metrics` 표 6행.
- 121줄: 학습 곡선입니다. 나무마다 잰 검증 MAE를 step 0~166으로 남겨 `metrics` 표 167행이 됩니다. `latest_metrics`에는 마지막 값(step 166, 1.8012) 1행만 들어갑니다.
- 합계는 `metrics` 173행, `latest_metrics` 7행입니다. 6단계에서 표를 직접 세어 확인합니다.

**⑧ 모델을 남긴다**: [`train.py:160-164`](train.py#L160-L164)

```python
return mlflow.lightgbm.log_model(model, name="model", input_example=feature_matrix(valid)[:5])
```

코드는 한 줄이지만 MLflow 안의 `Model.log`(`mlflow/models/model.py:1065`)에서 여섯 단계를 거칩니다. 4편에서는 이 가운데 ⑧-4를 뺀 다섯 단계가 서버 로그에 HTTP 요청으로 하나씩 찍힙니다. ⑧-4는 클라이언트 안에서만 일어나서, 그동안 서버 로그가 조용합니다.

| | MLflow 안의 코드 | 남는 것 |
|---|---|---|
| ⑧-1 | `_create_logged_model` (`model.py:1172`). 그 전에 run의 파라미터를 읽어 같이 넘긴다 | `logged_models` 표 1행, 상태 **PENDING**. `logged_model_params` 8행. id(`m-…`)가 이때 정해진다 |
| ⑧-2 | `client.log_outputs` (`model.py:1202`) | `inputs` 표에 run → 모델 연결 1행 |
| ⑧-3 | `log_model_metrics_for_step` (`model.py:1127`, 부르는 곳 1210). run의 지표 중 **step 0인 것만** 모델에 복사한다 | `logged_model_metrics` 7행. 5단계에서 이 "step 0만"이 화면에 드러난다 |
| ⑧-4 | `flavor.save_model` (`model.py:1228`). 임시 폴더에 파일 7개를 만든다 | `model.skops`, `MLmodel`, `requirements.txt` 등. 이 단계가 가장 오래 걸린다(6단계) |
| ⑧-5 | `client.log_model_artifacts` (`model.py:1308`) | 임시 폴더의 파일 7개를 `mlruns/1/models/m-…/artifacts/`로 복사 |
| ⑧-6 | `with _use_logged_model(...)` 블록을 나온다 (`model.py:1199`, 정의는 `mlflow/tracking/fluent.py:2564`) | 예외가 없었으면 **READY**, 있었으면 FAILED |

⑧-6의 코드는 이렇게 짧습니다. 4편에서 파일 올리기를 일부러 실패시킬 때 이 `except`가 실행됩니다.

```python
# mlflow/tracking/fluent.py:2564
def _use_logged_model(model):
    try:
        yield model
    except Exception:
        finalize_logged_model(model.model_id, LoggedModelStatus.FAILED)
        raise
    else:
        finalize_logged_model(model.model_id, LoggedModelStatus.READY)
```

**⑨ run을 닫는다**: `with` 블록 끝 → `ActiveRun.__exit__`(`mlflow/tracking/fluent.py:383`)

```python
status = RunStatus.FINISHED if exc_type is None else RunStatus.FAILED
end_run(RunStatus.to_string(status))
```

`runs` 행의 상태가 FINISHED로 바뀌고 끝난 시각이 들어갑니다. run의 상태를 정하는 코드는 이 두 줄뿐입니다. 7단계에서 이 두 줄이 실행되지 못하게 만들어 봅니다.

**⑩ 요약을 출력한다**: [`train.py:214-222`](train.py#L214-L222). JSON의 `run_id`는 ③에서, `model_id`는 ⑧-1에서 받은 id입니다.

### 관찰

**터미널**: 줄마다 위의 어느 지점에서 나왔는지 맞춰 봅니다. 순서도 위와 같습니다.

| 출력 | 나온 곳 |
|---|---|
| `Creating initial MLflow database tables...` | ② (mlflow.db를 처음 만들 때만) |
| `Experiment with name 'etth1-ot-6h' does not exist. Creating a new experiment.` | ② |
| `UserWarning: Hint: Inferred schema contains integer column(s)` 두 번 | ⑤ 99줄 |
| `Detected uv project: found uv.lock and pyproject.toml in /workspace` | ⑧-1. 모델에 "uv 프로젝트에서 왔다"고 표시할지 정할 때 |
| 같은 줄 한 번 더, `Exported 205 dependencies via uv` | ⑧-4. `requirements.txt`를 만들 때 |
| `Private package indexes detected in uv lockfile` | ⑧-4. `uv.lock`에 PyTorch 전용 패키지 저장소 주소가 있다는 안내 |
| `Failed to resolve installed pip version` | ⑧-4. 이미지의 가상환경을 uv로 만들어서 pip이 없다 |
| JSON: `experiment_id` 1, `run_id`, `model_id`(m-…), `best_iteration` 67, `mae` 1.7903 | ⑩ |

**파일**

```bash
ls -la mlflow.db mlruns/1/
ls mlruns/1/models/*/artifacts/
```

`mlruns/1/` 안에는 `models/` 폴더만 있고 run id 이름의 폴더는 없습니다. ⑧-5가 파일을 모델 폴더에만 썼고, run 자신의 산출물은 하나도 없기 때문입니다.

### 원리

- **기록 순서가 곧 "실패했을 때 무엇이 남는가"를 정합니다.** 파라미터(⑤)는 학습(⑥) 전에, 지표(⑦)는 학습 뒤에, 모델(⑧)은 맨 끝에 남습니다. 7단계에서 학습 도중에 멈추면 ⑤까지만 남는 것을 봅니다.
- **MLflow 3.x의 모델(LoggedModel)은 run의 산출물이 아니라 run과 나란한 별도 객체입니다.** ⑧-1에서 id를 먼저 받아 PENDING으로 두고, 파일을 다 쓴 뒤에 READY로 바꿉니다. 그래서 "모델 행은 있는데 파일이 없는" 상태가 생길 수 있고, 4편에서 그 상태를 만듭니다.
- 실험의 산출물 위치 `/workspace/mlruns/1`은 컨테이너 안의 **절대 경로**입니다. 다른 컴퓨터에서 이 `mlflow.db`를 열면 경로가 맞지 않습니다. 4편에서는 서버를 두어 이 문제를 풉니다.

## 2. UI에서 run 하나 따라가기

> 목표: UI의 칸 하나하나가 1단계의 어느 지점, 어느 표에서 오는지 연결한다.

**실행** (터미널 A, 계속 띄워 둡니다. 끌 때는 Ctrl+C)

```bash
bash practice/ch03/run.sh ui
```

### 코드가 지나가는 길

[`run.sh:21-25`](run.sh#L21-L25)

```bash
exec docker compose run --rm --no-deps -e MLFLOW_TRACKING_URI= \
  -p "127.0.0.1:${UI_PORT}:${UI_PORT}" \
  workspace mlflow ui --host 0.0.0.0 --port "${UI_PORT}" --workers 1
```

- `mlflow ui`는 웹 서버입니다. ①과 같은 이유로 주소가 비어 있으니 같은 `/workspace/mlflow.db`를 읽습니다. 화면은 이 서버가 표를 읽어 돌려준 값을 그릴 뿐이고, 화면에서 바꾼 것(등록, 별칭)도 같은 파일에 씁니다.
- `-p 127.0.0.1:5001:5001`은 컨테이너의 5001번 포트를 Windows의 127.0.0.1:5001로 엽니다. `--host 0.0.0.0`은 컨테이너 밖에서 들어오는 접속을 받겠다는 뜻입니다.
- `--workers 1`: 기본은 프로세스 4개(약 1GB)인데, 혼자 보는 용도라 1개(약 0.4GB)로 줄였습니다.

Windows 브라우저로 http://127.0.0.1:5001 을 엽니다.

### 관찰

**첫 화면**

1. "Welcome to MLflow" 화면이 뜹니다. 왼쪽 위에 `GenAI | Model training` 전환 단추가 있고, 처음에는 GenAI가 골라져 있습니다. **Model training**을 누르면 왼쪽 메뉴가 Home, Experiments, Model registry로 바뀝니다. (GenAI 모드에는 Model registry가 없고 Prompts, AI Gateway가 있습니다.)
2. 오른쪽 MLflow Assistant 패널은 X로 닫습니다. 위의 telemetry 안내도 X로 닫아도 됩니다.
3. **Experiments**를 누르면 실험이 두 개 보입니다. `etth1-ot-6h`와 `Default`입니다. Default(id 0)는 실험을 지정하지 않고 기록하면 들어가는 곳입니다. ②가 `etth1-ot-6h`를 만들었으므로 run은 그쪽에 있습니다.

**run 목록**: `etth1-ot-6h`를 누르면 왼쪽에 Runs, Models, Traces가 생기고 Runs 표가 나옵니다.

| 칸 | 이 PC에서 본 값 | 표 | 1단계 지점 |
|---|---|---|---|
| Run Name | `manual-lr0.05-leaves31` | `runs.name` | ③. 겹쳐도 되는 이름표일 뿐이고 진짜 식별자는 run id |
| Created | 초록 체크 아이콘 + 시각 | `runs.status`, `start_time` | ③에서 RUNNING, ⑨에서 FINISHED. 체크가 FINISHED |
| Dataset | `etth1-2016Q3+2016Q4+2017Q1 (afa39432)` `+1` | `datasets`, `inputs` | ⑤. 괄호 안은 지문(digest) 앞 8자리, `+1`은 검증 데이터 |
| Duration | 1분 안팎 | `end_time - start_time` | ③~⑨ |
| Source | `train.py` | `mlflow.source.name` 태그 | ③ |
| Models | `model` | `logged_models` | ⑧-1 |

**run 화면**: run 이름을 누르면 위에 탭 다섯 개가 있습니다. Overview, Model metrics, System metrics, Traces, Artifacts.

Overview 탭에서 칸마다 어디서 왔는지 짚어 봅니다.

- 왼쪽 **Metrics (7)**: `mae` 1.7903, `rmse` 2.3795, `naive_mae` 2.2298, `skill` 0.1971, `rows` 2184, `best_iteration` 67, `valid_l1_by_tree` 1.8012. ⑦에서 왔고, `latest_metrics` 표를 읽은 값입니다.
- 왼쪽 **Parameters (8)**: 하이퍼파라미터 6개와 `train_slices`, `valid_slices`. ⑤의 85~88줄입니다.
- 왼쪽 아래 **Logged models (1)**: Status Ready. ⑧-6에서 READY가 됐습니다.
- 오른쪽 **About this run**: Status Finished(⑨), Run ID(③), **Source**에 `train.py`, 브랜치 이름, 커밋 해시 7자리가 나란히 있습니다(③의 자동 태그).
- 오른쪽 **Datasets**: 학습 데이터와 `+1`(⑤). **Tags**: `task: timeseries`, `logging: manual`(④). **Registered models**: None(아직 등록 전).

**예측**: `valid_l1_by_tree`의 값 1.8012가 `mae` 1.7903보다 큰 이유는 무엇일까요? 답은 아래 원리에 있습니다.

**캡처 `mlops-run-overview.png`**: 이 Overview 화면을 찍습니다. 왼쪽의 Metrics와 Parameters, 오른쪽 About this run의 Source(커밋)가 한 화면에 들어오게 합니다. 이 한 장이 "run 하나 = 그때의 설정(⑤) + 결과(⑦) + 코드 버전(③)"을 보여 줍니다. 노트 03의 `#### 직접 기록` 절에서, run에 남는 것을 정리한 표 바로 아래에 넣습니다. 그 자리에는 아직 캡처 주석이 없으니 새로 넣습니다.

**Model metrics 탭**: 지표마다 차트가 하나씩 있습니다. `mae`처럼 step 0에 점이 하나뿐인 지표는 막대로 나옵니다. `valid_l1_by_tree`만 가로축이 Step(0~166)인 선으로 나옵니다. 이 차트는 `latest_metrics`가 아니라 `metrics` 표의 이력 전체를 읽습니다.

**Artifacts 탭**: 파일 7개가 보입니다. 위에 **"You're viewing artifacts assigned to a logged model associated with this run."**이라는 안내가 있습니다. `MLmodel`을 누르면 오른쪽에 내용이 보입니다. `signature`(입력 float32 [-1, 15])와 `run_id`를 찾아봅니다.

### 원리

- 화면은 표를 읽어 보여 줄 뿐입니다. Metrics는 `latest_metrics`(지표마다 마지막 값)를, 차트는 `metrics`(step 이력 전부)를 읽습니다. Parameters는 `params`, Datasets는 `datasets`와 `inputs`, Tags는 `tags` 표에서 옵니다.
- Tags에는 ④에서 직접 붙인 2개만 보입니다. ③의 자동 태그 7개는 화면에서 About this run의 Source, Created by 칸으로 모양을 바꿔 나옵니다. 원래 모양은 6단계에서 표를 직접 열어 확인합니다.
- **커밋 해시는 "지금 HEAD 커밋"일 뿐입니다.** `git_utils.py:71`은 `repo.head.commit.hexsha` 하나만 읽습니다. 커밋하지 않은 수정이 있어도 같은 해시가 찍히고, 수정이 있었다는 표시도 남지 않습니다. 03 확인 질문 1번의 답이 여기 있습니다. 11단계에서 직접 확인합니다.
- `valid_l1_by_tree`가 `mae`보다 큰 이유: 같은 이름의 지표를 step마다 여러 번 남기면 곡선이 되고, 표에는 마지막 값만 나옵니다. 조기 종료는 가장 좋았던 나무(67번째)에서 100그루를 더 가 본 뒤에 멈춥니다. 그래서 곡선의 마지막(167번째)은 최저점보다 높습니다. `mae`는 가장 좋았던 나무로 계산한 값입니다.
- Artifacts 탭은 이름이 "run의 산출물"인데 실제로는 run에 연결된(⑧-2) **모델**의 파일을 보여 줍니다. run 자신의 산출물 폴더는 비어 있습니다. 1단계에서 `mlruns/1/`에 run id 폴더가 없던 것과 같은 이야기이고, 위의 안내 문구가 그 사실을 말해 줍니다.

## 3. run을 더 쌓고 목록에서 비교하기

> 목표: 실험 추적의 쓸모, 즉 "여러 시도를 한 화면에서 비교하고 고르기"를 해 본다.

**예측**: learning_rate 0.02, 0.05, 0.1, 0.2 중 검증 MAE가 가장 낮은 것은 무엇일까요? 0.2처럼 큰 학습률은 몇 번째 나무에서 가장 좋을까요?

**실행** (터미널 B, 한 번에 30초~1분 30초)

```bash
for lr in 0.02 0.1 0.2; do
  bash practice/ch03/run.sh python practice/ch03/train.py --param learning_rate=$lr
done
bash practice/ch03/run.sh python practice/ch03/train.py --param num_leaves=63
```

### 코드가 지나가는 길

`--param learning_rate=0.1`이 어떻게 params 표의 `"0.1"`이 되는지 따라갑니다.

1. [`train.py:57-63`](train.py#L57-L63): `--param`은 `action="append"`라서 여러 번 줄 수 있고, `["learning_rate=0.1"]`처럼 문자열 목록이 됩니다.
2. [`train.py:178`](train.py#L178) → [`mlops/runner.py:66`](../../mlops/runner.py#L66) `parse_params`: `=`로 나누고, 기본값에 없는 이름이면 거부합니다(오타 방지).
3. [`mlops/runner.py:42-54`](../../mlops/runner.py#L42-L54) `_cast`: 기본값의 타입으로 바꿉니다. 기본 `learning_rate`가 `0.05`(float)이므로 `float("0.1")`이 됩니다.

   ```python
   return type(default)(raw)  # 기본값이 float이면 float("0.1") → 0.1
   ```

4. [`train.py:70-73`](train.py#L70-L73): run 이름이 `manual-lr0.1-leaves31`이 됩니다(③).
5. [`train.py:85`](train.py#L85): `log_params`가 0.1을 다시 **문자열** `"0.1"`로 바꿔 저장합니다(⑤). 학습에는 숫자가 쓰이고, 기록에는 글자가 남습니다.

이 PC에서 본 값:

| run | best_iteration | 검증 mae |
|---|---|---|
| `manual-lr0.02-leaves31` | 161 | 1.7996 |
| `manual-lr0.05-leaves31` | 67 | 1.7903 |
| `manual-lr0.1-leaves31` | 28 | 1.7965 |
| `manual-lr0.2-leaves31` | 28 | **1.7897** |
| `manual-lr0.05-leaves63` | 59 | 1.8361 |

### 관찰

**목록에 지표 칸 붙이기**: 브라우저에서 run 목록을 새로 고칩니다.

1. 표 위의 **Columns**를 누릅니다. Attributes, Metrics (7), Parameters (8), Tags (2) 묶음이 나옵니다. 괄호 안의 숫자는 ⑦, ⑤, ④에서 남긴 개수입니다.
2. 위의 Search columns 칸에 `mae`를 쳐서 Metrics 아래의 `mae`를 체크합니다. 같은 방법으로 Parameters 아래의 `learning_rate`, `num_leaves`를 체크합니다.
3. 표에 Metrics, Parameters 칸이 생깁니다. `mae` 칸 머리를 누르거나 **Sort**에서 `mae`를 골라 오름차순으로 정렬합니다.

**캡처 `mlops-runs.png`**: run 다섯 개가 mae 오름차순으로 정렬되고 learning_rate, num_leaves 칸이 함께 보이는 화면을 찍습니다. 이 화면이 실험 추적을 하는 이유 그 자체입니다. 다섯 번의 시도가 설정과 결과를 단 채 한 표에 모이고, 숫자 하나로 줄 세울 수 있습니다. 노트 03의 `#### 직접 기록` 절 끝에 있는 `<!-- 캡처 자리: ![[mlops-runs.png]] -->` 자리에 넣습니다.

**검색**: 표 위의 검색 칸(회색 예시 `metrics.rmse < 1 and params.model = "tree"`)에 차례로 넣어 봅니다.

| 검색어 | 이 PC 결과 |
|---|---|
| `metrics.mae < 1.795` | 2개(lr 0.05, 0.2) |
| `params.learning_rate = '0.2'` | 1개 |
| `params.learning_rate = 0.2` | 오류: `Expected a quoted string value for parameter` |
| `params.learning_rate > '0.1'` | 오류: `Invalid comparator '>'` (쓸 수 있는 것은 `=`, `!=`, `LIKE`, `ILIKE` 등) |

**학습 곡선 겹쳐 보기**: 표 위 왼쪽의 아이콘 세 개 중 **두 번째(꺾은선 모양)**를 누르면 차트 보기로 바뀝니다. 위의 Search metric charts 칸에 `valid_l1`을 치면 `valid_l1_by_tree` 차트 하나에 run별 곡선이 겹쳐 나옵니다. ⑦의 121줄이 남긴 곡선들입니다. 왼쪽 목록의 눈 아이콘으로 run을 숨기거나 다시 켤 수 있습니다.

- lr 0.2와 0.1은 30그루 안쪽에서 바닥을 찍고 **다시 올라갑니다**. 과적합입니다. 곡선은 바닥에서 100그루를 더 간 뒤에 끝납니다.
- lr 0.02는 천천히 내려가서 160그루 근처에서 바닥입니다.
- 바닥의 높이는 거의 같습니다(1.79~1.80). 학습률은 "얼마나 빨리 가는가"를 바꿨을 뿐, "어디까지 가는가"는 거의 바꾸지 못했습니다.

**캡처 `mlops-curves.png`**: 곡선이 겹친 차트를 찍습니다. lr 0.2의 반등과 lr 0.02의 느린 하강이 함께 보여야 합니다. 이 그림은 숫자 하나(mae)로는 안 보이는 것, 즉 "같은 결과에 어떻게 도달했는가"를 보여 줍니다. 노트 03 `#### 직접 기록` 절에서 `mlops-runs.png` 바로 다음에 넣습니다.

### 원리

- 파라미터는 `params` 표에 **문자열**로 들어갑니다(위 5번). 그래서 검색할 때 따옴표가 필요하고 크기 비교를 못 합니다. 지표는 숫자라서 `<`, `>`가 됩니다. 파라미터로 범위 검색이 필요하면 같은 값을 지표나 태그로도 남기는 수밖에 없습니다.
- 목록 화면이 빠른 것은 `latest_metrics` 표 덕분입니다. 곡선 점 수백 개가 든 `metrics` 표를 매번 훑지 않고, 지표마다 마지막 값 하나만 읽습니다. 차트 보기는 반대로 `metrics` 표의 step 이력을 읽습니다.
- 가장 낮은 mae(1.7897)와 두 번째(1.7903)의 차이는 0.0006입니다. 이 차이로 모델을 골라도 되는지는 10단계에서 다시 봅니다.

## 4. autolog run과 나란히 보기

> 목표: 자동 기록이 무엇을 대신해 주고 무엇을 놓치는지, 코드가 감싸는 지점을 보고 설명한다.

**예측**: autolog는 파라미터를 몇 개 남길까요? `mae`와 `skill`은 남을까요? 데이터 요약은 몇 개일까요?

**실행**

```bash
bash practice/ch03/run.sh python practice/ch03/train.py --autolog
```

### 코드가 지나가는 길

**autolog를 켠다**: [`train.py:186-191`](train.py#L186-L191)

```python
if args.autolog:
    mlflow.lightgbm.autolog()
```

이 한 줄은 아무것도 기록하지 않습니다. 대신 LightGBM의 함수를 MLflow가 만든 함수로 바꿔 끼웁니다(patch). `mlflow/lightgbm/__init__.py`에서 바꿔 끼우는 곳은 넷입니다.

| 줄 | 바꿔 끼우는 함수 | 그 함수가 불리면 남기는 것 |
|---|---|---|
| 1018 | `lightgbm.Dataset.__init__` | 학습·검증 데이터를 만들 때마다 데이터 요약 |
| 1019-1026 | `lightgbm.train` | `lightgbm.train`을 직접 부르는 코드용. 이 과제에서는 쓰이지 않는다 |
| 1033-1040 | `lightgbm.sklearn.train` | 파라미터, 나무마다의 검증 지표, feature importance 그림과 json. 모델은 여기서 남기지 않는다 |
| 1045 | `mlflow.sklearn._autolog(...)`로 `LGBMRegressor.fit` | fit이 끝난 뒤의 모델(LoggedModel) |

**학습한다**: [`train.py:198`](train.py#L198)에서 run은 우리가 엽니다. autolog는 열려 있는 run에 기록합니다. [`train.py:201`](train.py#L201)의 `if not args.autolog`가 ⑤를 건너뜁니다. 그다음 ⑥의 [`timeseries.py:168`](../../mlops/tasks/timeseries.py#L168) `model.fit(...)`이 불리는데, 이제 이것은 바꿔 끼운 함수입니다. fit 안에서 LightGBM이 다시 `lightgbm.sklearn.train`을 부르고, 그것도 바꿔 끼운 함수입니다.

```python
# mlops/tasks/timeseries.py:156-173 (발췌). 이름이 바뀌는 곳을 보세요
model = lgb.LGBMRegressor(
    n_estimators=params["n_estimators"],     # LightGBM 안에서 train(num_boost_round=...)으로 넘어간다
    random_state=params["seed"],             # 우리 이름 seed → LightGBM 이름 random_state
    ...
)
model.fit(..., callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)])
```

- autolog는 LightGBM이 **받은** 값을 남깁니다. 그래서 이름이 `num_boost_round`, `random_state`이고, 우리가 정하지 않은 기본값(`boosting_type`, `colsample_bytree`, `reg_alpha` …)까지 들어가 22개가 됩니다.
- `early_stopping_rounds`는 파라미터가 아니라 콜백 객체로 넘어갔습니다. autolog는 이 값을 파라미터로 보지 못합니다.
- 지표는 나무마다 잰 `valid_0-l1`(step 0~166)입니다. 끝에 가장 좋았던 나무의 값을 step 167로 한 번 더 남기고, `best_iteration`, `stopped_iteration`을 남깁니다.

**나머지를 건너뛴다**: [`train.py:207`](train.py#L207)의 `if not args.autolog`가 ⑦과 ⑧을 건너뜁니다. 그래서 `mae`, `skill`, `rmse`가 없고, 마지막 JSON의 `model_id`가 `null`입니다. 모델은 autolog가 따로 남겼으니 이 스크립트는 그 id를 모릅니다. 끝으로 [`train.py:209-212`](train.py#L209-L212)의 `finally`가 autolog를 끕니다. 끄지 않으면 같은 프로세스의 다음 학습도 계속 기록됩니다.

### 관찰

**autolog run의 Overview**

- **Metrics (3)**: `valid_0-l1` 1.79034, `stopped_iteration` 167, `best_iteration` 67. `mae`, `skill`, `rmse`는 없습니다.
- **Parameters (22)**: `boosting_type`, `colsample_bytree`, `max_depth`, `reg_alpha` …. `train_slices`는 없습니다.
- **Datasets**: `dataset (10f4cd40)` `Train` `+2`. 이름이 모두 `dataset`이고 세 개입니다.
- Artifacts 탭: `feature_importance_gain.json/.png`, `feature_importance_split.json/.png` 4개. 이번에는 run 자신의 산출물입니다. `.png`를 누르면 그림이 보입니다.

**비교 화면**

1. Runs 목록(표 보기)에서 `autolog-lr0.05-leaves31`과 `manual-lr0.05-leaves31`의 왼쪽 체크박스를 체크합니다.
2. 표 위에 Rename, Delete, **Compare**, Add tags 단추가 나타납니다. **Compare**를 누릅니다.
3. "Comparing 2 Runs from 1 Experiment" 화면입니다. 위에서부터 Visualizations, Run details, Parameters, Metrics, Artifacts, Tags 순서입니다. Parameters와 Metrics의 **Show diff only**를 켜면 다른 줄만 남습니다.

| 볼 줄 | autolog | manual | 코드에서 본 이유 |
|---|---|---|---|
| `n_estimators` / `num_boost_round` | 빈칸 / 2000 | 2000 / 빈칸 | 같은 설정, 다른 이름(`timeseries.py:157`) |
| `seed` / `random_state` | 빈칸 / 42 | 42 / 빈칸 | 같은 설정, 다른 이름(`timeseries.py:161`) |
| `early_stopping_rounds` | 빈칸 | 100 | 콜백으로 넘겨서 autolog가 못 본다(`timeseries.py:173`) |
| `train_slices` | 빈칸 | 2016Q3,2016Q4,2017Q1 | 과제 정보는 LightGBM에 넘어가지 않는다(⑤ 86~88줄) |
| `mae`, `skill` | NaN | 1.79, 0.197 | fit 밖에서 계산한 값(`train.py:172`) |
| `valid_0-l1` / `valid_l1_by_tree` | 1.79 / NaN | NaN / 1.801 | 같은 곡선, 다른 이름과 다른 마지막 값 |

Artifacts 칸은 이 버전에서 "Something went wrong with this section" 오류가 날 수 있습니다. 비교에는 상관없습니다.

**캡처 `mlops-compare.png`**: Show diff only를 켠 Parameters와 Metrics 표를 찍습니다. 이름만 다른 짝(`n_estimators`/`num_boost_round`)과 NaN이 함께 보여야 합니다. 이 화면이 "autolog는 함수 경계 안쪽만 본다"는 증거입니다. 노트 03의 `#### autolog와 비교` 절 끝의 `<!-- 캡처 자리: ![[mlops-compare.png]] -->` 자리에 넣습니다.

### 원리

- autolog는 LightGBM 함수를 감싸서(patch) **그 함수로 들어가고 나오는 것만** 기록합니다. 들어간 설정은 기본값까지 다 남기지만, 함수 밖에서 계산한 `mae`, `skill`과 과제 정보(어느 분기로 학습했나)는 모릅니다.
- 두 방식을 섞으면 같은 설정이 다른 이름으로 남습니다. `params.n_estimators = '2000'`으로 검색하면 autolog run이 빠집니다. 팀에서 기록 방식을 하나로 정해야 하는 이유입니다.
- `valid_0-l1`의 마지막 값(1.79034)이 `mae`와 같은 이유는 autolog가 가장 좋았던 나무의 값을 곡선 끝에 한 번 더 붙이기 때문입니다. 직접 남긴 곡선은 멈춘 시점에서 끝나서 1.8012입니다. (변화량의 MAE가 곧 온도의 MAE입니다. 예측과 정답에 같은 "지금 온도"를 더하기 때문입니다.)
- 데이터 요약 세 개는 학습 세트 하나와 검증 세트 기록 두 개입니다. 두 검증 기록의 지문이 서로 다른 이유는 확인하지 못했습니다.
- 두 run의 모델은 같은 모델입니다. 설정과 시드가 같아서 `model.skops` 크기도 같습니다(6단계). 기록하는 방법만 다릅니다.

## 5. 모델 화면 보기

> 목표: MLflow 3.x에서 run과 모델이 별개의 객체라는 것을 화면으로 확인하고, 모델에 붙은 지표가 어디서 왔는지 안다.

이 단계는 새 코드를 돌리지 않습니다. 1단계의 ⑧이 만든 것을 봅니다.

### 관찰

1. 실험 왼쪽 메뉴의 **Models**를 누릅니다. 6줄이 보이고 Model name이 **전부 `model`**입니다. Status Ready, Logged from `train.py`, Source run에 run 이름이 있습니다. 오른쪽으로 스크롤하면 `mae` 칸이 있고, autolog 모델만 `-`입니다.
2. Source run이 `manual-lr0.05-leaves31`인 `model`을 누릅니다. 제목 옆에 **Not registered**, 오른쪽 위에 **Register model** 단추가 있습니다. 탭은 Overview, Traces, Artifacts입니다.
3. Overview의 **About this logged model**: Status Ready(⑧-6), Model ID(`m-…`, ⑧-1), Source run과 Source run ID(⑧-1의 `source_run_id`), Logged from(파일, 브랜치, 커밋). **Datasets used**는 None, **Model versions**도 None입니다.
4. 아래 **Metrics (7)** 표의 Metric, Dataset, Source run, Value 칸에서 `valid_l1_by_tree`의 Value를 찾습니다. 이 PC에서 **2.199**였습니다. 2단계에서 본 run 화면의 같은 지표는 1.8012였습니다.

**예측**: 같은 run, 같은 이름의 지표인데 왜 값이 다를까요?

### 코드로 풀기

⑧-3의 `log_model_metrics_for_step`(`mlflow/models/model.py:1127-1146`)이 run의 지표를 모델에 복사할 때 거르는 조건이 이렇습니다.

```python
for metric in history                                   # 지표 하나의 이력 전체
if metric.step == step and metric.model_id is None      # step이 같은 것만
```

`step`은 `log_model`의 인자이고 기본값이 0입니다(`step: int = 0`). 그래서 step이 0 하나뿐인 `mae` 같은 지표는 그대로 복사되지만, 곡선 지표는 **step 0의 값, 즉 나무 한 그루일 때의 MAE**(2.199)가 복사됩니다. 가장 좋았던 값도 마지막 값도 아닙니다. 표에서 직접 확인합니다.

```bash
bash practice/ch03/run.sh python - <<'EOF'
import sqlite3
db = sqlite3.connect("file:mlflow.db?mode=ro", uri=True)
run = "manual-lr0.05-leaves31"
print("run의 곡선 (step, 값):", db.execute(
    "SELECT m.step, round(m.value, 4) FROM metrics m JOIN runs r ON r.run_uuid = m.run_uuid"
    " WHERE r.name = ? AND m.key = 'valid_l1_by_tree' AND m.step IN (0, 166) ORDER BY m.step",
    (run,)).fetchall())
print("모델에 복사된 값 (step, 값):", db.execute(
    "SELECT lm.metric_step, round(lm.metric_value, 4) FROM logged_model_metrics lm"
    " JOIN runs r ON r.run_uuid = lm.run_id WHERE r.name = ? AND lm.metric_name = 'valid_l1_by_tree'",
    (run,)).fetchall())
EOF
```

```text
run의 곡선 (step, 값): [(0, 2.199), (166, 1.8012)]
모델에 복사된 값 (step, 값): [(0, 2.199)]
```

### 원리

- 모델 이름이 모두 `model`인 것은 `log_model(name="model")`의 이름이 "그 run 안에서 부르는 이름"이기 때문입니다. 모델의 정체는 `m-…` id입니다. 사람이 부를 이름(`etth1-ot`)은 레지스트리에서 따로 붙입니다(8단계).
- 모델 화면의 지표와 파라미터는 run 것의 **복사본**입니다(⑧-1, ⑧-3). 그래서 run을 거치지 않고 모델끼리 비교할 수 있습니다. 다만 무엇을 복사할지는 `log_model(step=...)`이 정합니다. 곡선 지표로 모델을 줄 세우면 "나무 한 그루일 때"끼리 비교하게 됩니다. 모델끼리 비교할 지표는 `mae`처럼 step 0에 하나로 정리해 남기는 편이 안전합니다.
- Datasets used가 None인 것은 ⑤의 `log_input`이 데이터를 run에만 연결했기 때문입니다. 모델에서 데이터까지 가려면 Source run을 한 번 거쳐야 합니다.
- autolog 모델의 `mae`가 `-`인 것은 4단계와 같은 이유입니다. 모델 목록에서 mae로 정렬해 고르는 방식이라면 autolog 모델은 후보에서 빠집니다.

## 6. 저장소 들여다보기

> 목표: UI가 보여 준 모든 것이 "SQLite 표 몇 개와 파일 몇 개"라는 것을 직접 열어서 확인한다.

**실행**. `M=` 뒤에는 1단계 JSON의 `model_id`(lr 0.05 run의 모델)를 넣습니다.

```bash
bash practice/ch03/run.sh python practice/ch03/inspect_store.py --run-name manual-lr0.05-leaves31
M=m-…   # 1단계 JSON의 model_id
bash practice/ch03/run.sh bash -c "head -40 mlruns/1/models/$M/artifacts/MLmodel"
bash practice/ch03/run.sh bash -c "wc -l mlruns/1/models/$M/artifacts/requirements.txt; head -4 mlruns/1/models/$M/artifacts/requirements.txt"
```

### 코드가 지나가는 길

[`inspect_store.py`](inspect_store.py)는 MLflow API를 거치지 않고 SQLite 파일과 폴더를 직접 읽습니다.

1. [`main`, 173줄](inspect_store.py#L173) → [`sqlite_path`, 77-83줄](inspect_store.py#L77-L83): `mlflow.get_tracking_uri()`가 돌려준 `sqlite:///mlflow.db`에서 파일 경로를 꺼냅니다. ①과 같은 규칙으로 정해진 주소입니다.
2. [`connect_read_only`, 72-74줄](inspect_store.py#L72-L74): `mode=ro`로 엽니다. UI가 같은 파일을 쓰고 있어도 건드리지 않습니다.
3. [`show_tables`, 86줄](inspect_store.py#L86): `sqlite_master`에서 표 이름을 모두 읽고 행 수를 셉니다. 표마다의 설명은 [`TABLE_NOTES`, 20줄](inspect_store.py#L20)에 있습니다.
4. [`show_run_counts`, 115줄](inspect_store.py#L115): run마다 `params`, `metrics`, `latest_metrics`, `tags` 표의 행 수를 셉니다([`RUN_COUNTS_SQL`, 43줄](inspect_store.py#L43)).
5. [`show_run_tags`, 130줄](inspect_store.py#L130): `--run-name`으로 고른 run의 태그를 표에 든 그대로 보여 줍니다. 이름을 주지 않으면 가장 최근 run입니다.
6. [`show_tree`, 152줄](inspect_store.py#L152): `mlruns/` 폴더를 나무 모양으로 보여 줍니다.

### 관찰

**표 목록**: `runs`, `params`, `metrics`, `latest_metrics`, `tags`, `datasets`, `inputs`, `logged_models` 등이 나옵니다. 그 밖에 비어 있는 표가 40개 넘게 있다는 줄이 나옵니다(트레이스, 프롬프트 같은 GenAI 기능용).

**run별 행 수** (이 PC에서 본 두 줄)

```text
== run별 행 수 (metrics: 지표 기록 전부, latest: 지표마다 마지막 값 하나)
  run 이름                         상태      params  metrics  latest  tags
  manual-lr0.05-leaves31           FINISHED       8      173       7     9
  autolog-lr0.05-leaves31          FINISHED      22      170       3     9
```

- manual: params 8(⑤), metrics 173 = 지표 6 + 곡선 167(⑦), latest 7, tags 9 = 자동 7(③) + 직접 2(④). 1단계에서 센 그대로입니다.
- autolog: params 22, metrics 170 = `valid_0-l1` 168(step 0~167) + `best_iteration` + `stopped_iteration`, latest 3. 4단계에서 화면으로 본 것과 같습니다.
- 나머지 manual run은 곡선 길이만큼 metrics 행 수가 다릅니다. 곡선 길이는 best_iteration + 100입니다.

**태그 원본**: UI의 Tags에는 2개만 보였지만 표에는 9개가 있습니다.

```text
== run manual-lr0.05-leaves31 (FINISHED, …)의 태그
  logging                    manual
  mlflow.runName             manual-lr0.05-leaves31
  mlflow.source.git.branch   practice/ch03-ch04
  mlflow.source.git.commit   0965967891662413b7edf95bfdafaefad9e260f6
  mlflow.source.git.repoURL  git@github.com:Utaewook/mlops.git
  mlflow.source.name         practice/ch03/train.py
  mlflow.source.type         LOCAL
  mlflow.user                …
  task                       timeseries
```

**폴더**: autolog run만 `mlruns/1/<run id>/artifacts/`에 feature importance 4개가 있습니다. 모델은 모두 `models/m-…/artifacts/`에 있습니다. autolog 모델 폴더에는 파일이 5개이고 `input_example.json`, `serving_input_example.json`이 없습니다. autolog는 기본으로 입력 예시를 남기지 않기 때문입니다.

**`MLmodel`**: `flavors`에 `lightgbm`과 `python_function` 두 개, `signature`의 입력은 `tensor`, `float32`, `[-1, 15]`이고 `run_id`, `model_id`가 있습니다. `artifact_path`는 `/workspace/mlruns/…` 절대 경로입니다.

**`requirements.txt`**: **205줄**입니다. 첫 줄이 `--extra-index-url https://download.pytorch.org/whl/cpu`이고, `accelerate`, `aiohttp`, `alembic` … 순서로 이 저장소의 의존성이 전부 들어 있습니다.

### 원리

- 1단계 터미널의 `Exported 205 dependencies via uv`가 이 205줄입니다. MLflow 3.17은 작업 폴더에 `uv.lock`과 `pyproject.toml`이 있으면 모델의 환경을 `uv export`로 적습니다(⑧-4). 이 저장소 전체(torch, dagster 포함)의 의존성이라 LightGBM 모델 하나에는 지나치게 많습니다. 이 목록으로 서빙 환경을 만들면 이미지가 수 GB가 됩니다.
- `MLFLOW_UV_AUTO_DETECT=false`로 끄면 모델을 한 번 불러 보면서 import된 패키지를 적습니다. 이 PC에서는 12줄이었는데, 그래도 쓰지 않는 torch와 pytest가 섞였습니다.
- 정확하게 하려면 `log_model(..., pip_requirements=[...])`로 직접 적습니다. **환경 기록은 저절로 정확해지지 않습니다.** 모델을 등록하기 전에 `requirements.txt`를 한 번 열어 보는 습관이 필요합니다.

```bash
# 직접 비교해 보기 (run이 하나 더 생긴다)
bash practice/ch03/run.sh env MLFLOW_UV_AUTO_DETECT=false python practice/ch03/train.py --param learning_rate=0.4
```

## 7. 부숴 보기: 학습 도중에 멈추면

> 목표: run의 상태(RUNNING, FINISHED, FAILED)를 누가 어떻게 정하는지 알고, 그 상태를 얼마나 믿을 수 있는지 판단한다.

**예측**: 학습 도중에 Ctrl+C를 누르면 run은 어떤 상태로 남을까요? 컨테이너가 강제로 죽으면(메모리 부족으로 죽는 경우와 같습니다) 어떻게 될까요? 각각 파라미터, 지표, 모델 중 무엇이 남을까요?

**실행 A: Ctrl+C**. 학습이 오래 걸리도록 학습률을 낮춥니다.

```bash
bash practice/ch03/run.sh python practice/ch03/train.py --param learning_rate=0.01
```

`Hint: Inferred schema contains integer column(s)` 경고가 **두 번째로** 뜨면 바로 Ctrl+C를 누릅니다. ⑤까지 끝났고 ⑥(학습)이 시작된 시점입니다. 터미널에 `KeyboardInterrupt`가 나옵니다.

**실행 B: 강제 종료**. 같은 명령을 다시 돌리고, 경고가 두 번 뜨면 **새 터미널**에서 학습 컨테이너만 골라 죽입니다. UI 컨테이너도 같은 이미지로 떠 있으므로 명령에 `train.py`가 들어간 컨테이너만 고릅니다.

```bash
docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep 'train.py' | cut -d' ' -f1 | xargs -r docker kill
```

### 코드가 지나가는 길

**Ctrl+C**: 터미널이 컨테이너의 파이썬 프로세스에 SIGINT를 보냅니다. 파이썬은 SIGINT를 `KeyboardInterrupt` 예외로 바꿔, 그때 실행 중이던 곳(⑥의 `model.fit` 안)에서 던집니다. 예외는 `with` 블록을 빠져나가며 ⑨의 `ActiveRun.__exit__`를 부릅니다.

```python
# mlflow/tracking/fluent.py:383
def __exit__(self, exc_type, exc_val, exc_tb):
    ...
    # exc_type = KeyboardInterrupt
    status = RunStatus.FINISHED if exc_type is None else RunStatus.FAILED
    end_run(RunStatus.to_string(status))  # → runs 행이 FAILED
    return exc_type is None  # False → 예외를 다시 던진다
```

마지막 줄이 `False`를 돌려주므로 예외는 삼켜지지 않고 터미널까지 올라옵니다. 그래서 `KeyboardInterrupt` 추적 메시지가 보입니다.

**docker kill**: Docker는 SIGKILL을 보냅니다. SIGKILL은 프로세스가 가로챌 수 없고 그 즉시 끝납니다. 예외도 생기지 않으므로 `__exit__`가 불리지 않습니다. ⑨가 없으니 `runs` 행은 ③에서 쓴 RUNNING 그대로 남습니다.

### 관찰

run 목록을 새로 고칩니다.

| | Ctrl+C | 강제 종료 |
|---|---|---|
| Created 칸 아이콘 | 빨간 X (Failed) | 시계 (Running) |
| Duration | 있음 | 비어 있음(⑨의 끝난 시각이 없다) |
| Parameters | 8개(⑤까지 남았다) | 8개 |
| Metrics, Models | 없음(⑦, ⑧에 못 갔다) | 없음 |
| 시간이 지나면 | 그대로 | **계속 Running** |

**캡처 `mlops-run-states.png`**: run 목록에서 체크(Finished), X(Failed), 시계(Running) 아이콘이 함께 보이는 화면을 찍습니다. 이 화면이 보여 주는 것은 "Running이 지금 돌고 있다는 뜻이 아니다"입니다. 시계 아이콘의 run은 이미 죽었습니다. 노트 03에는 이 내용의 자리가 아직 없습니다. `#### 직접 기록` 다음에 "run 상태는 믿을 수 있나" 같은 절을 새로 만들어 넣습니다.

### 원리

- run의 상태는 서버나 MLflow가 지켜보다가 정하는 것이 아닙니다. **학습하는 프로세스가 스스로 보고**합니다. 보고하는 코드는 위의 `__exit__` 두 줄뿐입니다.
- SIGKILL은 프로세스에게 아무 기회도 주지 않습니다. 끝났다고 보고할 주체가 사라졌으니 run은 영원히 RUNNING입니다. MLflow 오픈소스에는 "살아 있으면 주기적으로 신호를 보내는" 장치(heartbeat)가 없습니다.
- 그래서 **RUNNING은 "돌고 있다"가 아니라 "끝났다는 보고를 못 받았다"**입니다. 운영에서는 오래된 RUNNING run을 찾아 `MlflowClient().set_terminated(run_id, "KILLED")`로 정리하는 작업을 둡니다.
- 파라미터는 남고 지표와 모델이 없는 것은 1단계의 기록 순서 때문입니다(⑤ → ⑥ → ⑦ → ⑧). 무엇을 시도하다 죽었는지는 남습니다.
- 코드로 한 걸음 더: ⑧-6의 `_use_logged_model`은 `except Exception`만 잡습니다. `KeyboardInterrupt`는 `Exception`이 아니라 `BaseException`의 자식입니다. 그러니 ⑧을 하는 도중(`Exported 205 dependencies` 무렵)에 Ctrl+C를 누르면, run은 `__exit__` 덕분에 FAILED가 되지만 모델은 FAILED로 바뀌지 못하고 **PENDING**에 남을 것입니다. 코드를 읽고 내린 추론이라 이 PC에서 돌려 보지는 않았습니다. 해 보고 5단계의 Models 화면에서 Status를 확인해 보세요.
- `register.py`가 `attributes.status = 'FINISHED'`인 run만 후보로 고르는 이유가 이것입니다(8단계). 4편에서는 지표는 있는데 모델 파일이 없는 더 나쁜 경우("반쪽 기록")를 만듭니다.

정리: 이 run들은 목록에서 체크하고 Delete로 지웁니다. 지운 run은 사라지지 않습니다. **State: Active**를 **Deleted**로 바꾸면 다시 보입니다. 삭제 표시만 되는 것이고, 4편에서 자세히 봅니다.

---

# 실습 2. 모델 레지스트리

## 8. CLI로 등록하기 (버전 1)

> 목표: 등록이 무엇을 바꾸고 무엇을 바꾸지 않는지 본다.

**예측**: 등록하고 나면 `predict.py`가 바로 새 모델로 예측할까요? 등록하면 모델 파일이 어딘가로 복사될까요?

**실행**

```bash
R="bash practice/ch03/run.sh python practice/ch03/register.py"
$R register
bash practice/ch03/run.sh python practice/ch03/predict.py
```

### 코드가 지나가는 길

`register.py register` → [`main`, 153줄](register.py#L153) → [`register`, 86줄](register.py#L86). 조회 두 번과 쓰기 한 번입니다.

**1. 어느 run을 등록할지 고른다**: [`best_run_id`, 43-63줄](register.py#L43-L63)

```python
runs = mlflow.search_runs(
    experiment_names=[experiment],
    filter_string="tags.logging = 'manual' and attributes.status = 'FINISHED'",  # ④의 태그, ⑨의 상태
    order_by=["metrics.mae ASC"],  # ⑦의 지표
    max_results=1,
)
```

조건마다 앞 단계에서 만든 것을 거릅니다. autolog run(mae가 없다, 4단계)과 7단계의 FAILED, RUNNING run이 빠지고, 남은 manual run 중 mae가 가장 낮은 `manual-lr0.2-leaves31`(1.7897)이 뽑힙니다.

**2. 그 run이 남긴 모델을 찾는다**: [`model_uri_of`, 66-83줄](register.py#L66-L83). MLflow 3.x에서 모델은 run의 산출물이 아니므로 `search_logged_models(filter_string="source_run_id = '…'")`로 찾습니다. ⑧-1에서 `logged_models` 행에 적힌 `source_run_id`를 거꾸로 검색하는 것입니다. 결과는 `models:/m-…` 주소입니다.

**3. 등록한다**: [`register.py:93`](register.py#L93)

```python
version = mlflow.register_model(model_uri, model_name)  # "models:/m-…", "etth1-ot"
```

- `registered_models` 표: `etth1-ot`이 없으므로 1행을 만듭니다. 터미널의 `Successfully registered model 'etth1-ot'.`
- `model_versions` 표: 버전 1 행. 모델 주소(`source = models:/m-…`)와 `run_id`를 함께 적습니다. 터미널의 `Created version '1' of model 'etth1-ot'.`
- 모델 쪽에도 표시를 남깁니다. `logged_model_tags`에 `mlflow.modelVersions` 태그가 붙습니다. 모델 파일은 복사하지도 옮기지도 않습니다. `mlruns/1/models/m-…/artifacts/`의 파일은 여전히 7개입니다.

**predict.py가 실패하는 곳**: [`predict.py:37`](predict.py#L37)

```python
version = MlflowClient().get_model_version_by_alias(model_name, alias)  # "etth1-ot", "champion"
```

`registered_model_aliases` 표에 `champion` 행이 아직 없으므로 `MlflowException: Registered model alias champion not found.`가 납니다.

### 관찰

**터미널**

```text
가장 좋은 run: manual-lr0.2-leaves31 (…), mae=1.7897
Successfully registered model 'etth1-ot'.
Created version '1' of model 'etth1-ot'.
등록: etth1-ot 버전 1 ← models:/m-… (run …)
별칭은 바뀌지 않았다. 쓰려면: register.py alias 1
```

**UI**

1. 왼쪽 메뉴 **Model registry**: Registered Models 표에 `etth1-ot`, Latest version `Version 1`이 있고 Aliased versions는 비어 있습니다.
2. `etth1-ot`을 누르면 Versions 표가 나옵니다(Version, Registered at, Created by, Tags, Aliases, Description). Aliases 칸에는 `Add`만 있습니다.
3. `Version 1`을 누르면 버전 화면입니다. Source Run(`manual-lr0.2-leaves31`), Source Model(`model`), Aliases(`Add`), **Stage (deprecated)**: None이 있고, 아래에 Schema(Inputs 1, Outputs 1)가 있습니다. Schema는 모델의 `MLmodel`에 적힌 signature입니다.
4. 실험의 Runs 목록으로 돌아가면 `manual-lr0.2-leaves31`의 Models 칸이 `etth1-ot v1`로 바뀌어 있습니다. 화면이 모델의 `mlflow.modelVersions` 태그를 읽어 보여 준 것입니다.

### 원리

- 등록은 `registered_models`에 이름을 만들고 `model_versions`에 행 하나(버전 1 → `models:/m-…`, run id)를 넣는 일입니다. 버전은 이미 있는 LoggedModel을 가리키는 **포인터**입니다.
- **등록과 별칭은 별개입니다.** 등록은 "이런 후보가 있다"는 기록이고, 별칭을 옮기는 것이 "이것을 쓰겠다"는 결정입니다. 그래서 등록만으로는 모델을 쓰는 쪽(`predict.py`)에 아무 일도 일어나지 않습니다. 새 버전을 등록할 때마다 운영 모델이 바뀐다면, 검증 없이 배포되는 셈입니다.
- Stage(Staging, Production)는 예전 방식입니다. 한 모델에 단계가 넷으로 고정되어 있고 이름을 바꿀 수 없어서, 이름을 자유롭게 짓는 별칭으로 바뀌었습니다.

## 9. UI로 등록하기 (버전 2)

> 목표: 같은 일을 UI로 해서 노트의 "UI로 등록해 본 과정" 자리를 채운다.

**실행** (UI)

1. Experiments → `etth1-ot-6h` → 왼쪽 **Models**.
2. Source run이 `manual-lr0.05-leaves31`인 `model`을 누릅니다.
3. 오른쪽 위 **Register model**을 누르면 "Register model" 대화상자가 뜹니다.
4. Model 칸(`Select a model`)을 누르면 `+ Create New Model`과, 그 아래 Models 목록(`etth1-ot`)이 나옵니다. **`etth1-ot`**을 고릅니다. 새 이름으로 만들려면 Create New Model을 고릅니다.
5. **Register**를 누르면 아래에 "Model registered successfully"가 뜨고, 오른쪽 About의 Model versions에 `etth1-ot v2`가 생깁니다.

### 코드가 지나가는 길

브라우저는 터미널 A의 `mlflow ui`(2단계)에 REST 요청을 보냅니다. 서버는 8단계의 `register_model`이 하는 것과 같은 처리로 `model_versions`에 버전 2 행을 넣습니다. 같은 `mlflow.db`에 쓰므로 CLI로 `$R list`를 돌려도 버전 2가 보입니다. 다른 점은 "어느 모델을 등록할지"를 코드의 검색 조건(8단계의 1번)이 아니라 사람이 눈으로 골랐다는 것뿐입니다.

### 관찰

**캡처 `mlops-register-ui.png`**: 4번에서 목록이 펼쳐진 대화상자를 찍습니다. `+ Create New Model`과 기존 `etth1-ot`이 함께 보여야 합니다. 이 화면은 "버전은 이름 아래에 쌓인다"는 것, 즉 새 모델을 기존 이름의 다음 버전으로 넣을지 새 이름을 만들지 고르는 자리를 보여 줍니다. 노트 03 `### 실습 2`의 `<!-- UI로 등록해 본 과정을 여기에 적는다 -->` 자리에 위 1~5번 과정과 함께 넣습니다.

### 원리

운영에서 사람이 UI로 등록하면 "왜 이 모델인가"가 남지 않습니다. 8단계는 기준(`tags.logging = 'manual'`, FINISHED, mae 최소)이 코드에 있어서 나중에 다시 따져 볼 수 있습니다. 그래서 등록은 스크립트나 파이프라인이 하는 편이 낫습니다.

## 10. 별칭 붙이고 옮기기

> 목표: 코드를 고치지 않고 운영 모델을 바꾸고 되돌린다. 그리고 그 기록이 어디에 남는지(남지 않는지) 확인한다.

**실행 1: UI로 champion 붙이기**

1. Model registry → `etth1-ot` → `Version 1`.
2. Aliases의 **Add**를 누르면 "Add/Edit alias for model version 1" 대화상자가 뜹니다.
3. `champion`을 치고 Enter, **Save aliases**를 누르면 Aliases가 `@ champion`이 됩니다.

```bash
bash practice/ch03/run.sh python practice/ch03/predict.py
```

**예측**: 이제 champion을 버전 2로 옮기면 2017Q3 MAE가 좋아질까요, 나빠질까요? 검증 MAE는 버전 1(1.7897)이 버전 2(1.7903)보다 조금 좋았습니다.

**실행 2: CLI로 옮기기**

```bash
$R alias 2                                          # etth1-ot@champion: 버전 1 → 버전 2
bash practice/ch03/run.sh python practice/ch03/predict.py
$R list
```

### 코드가 지나가는 길

**별칭 옮기기**: `register.py alias 2` → [`set_alias`, 98-109줄](register.py#L98-L109)

```python
current = client.get_registered_model(model_name).aliases.get(alias)  # 102: 지금 어느 버전인가
client.set_registered_model_alias(model_name, alias, version)  # 108: 옮긴다
```

`registered_model_aliases` 표에서 (`etth1-ot`, `champion`) 행의 버전 숫자가 1에서 2로 바뀝니다. 바뀌는 것은 이 숫자 하나입니다.

**예측하기**: `predict.py` → [`main`, 74줄](predict.py#L74) → [`load_by_alias`, 32-50줄](predict.py#L32-L50)

1. [37줄](predict.py#L37): `get_model_version_by_alias`가 별칭 표에서 버전 번호를 찾습니다. 돌려받은 버전 행에 `run_id`와 `source`(`models:/m-…`)가 들어 있어서 첫 줄에 그대로 찍힙니다.
2. [50줄](predict.py#L50): `mlflow.pyfunc.load_model("models:/etth1-ot/2")`. 버전 → `source`의 LoggedModel → `mlruns/1/models/m-…/artifacts/` 순서로 따라가서 `MLmodel`을 읽습니다. `flavors.python_function`에 적힌 대로 `mlflow.lightgbm`이 `model.skops`를 되살리고, signature(입력 float32 [-1, 15])로 입력을 검사하는 껍데기(pyfunc)를 씌웁니다.

   이때 MLflow는 모델 폴더에 `registered_model_meta`라는 두 줄짜리 파일(`model_name: etth1-ot`, `model_version: '2'`)을 써 넣습니다(`mlflow/store/artifact/models_artifact_repo.py:254`). 원래는 "내려받은 사본"에 이름과 버전을 적어 두는 기능입니다. 그런데 로컬 저장소는 파일을 복사하지 않고 원래 폴더 경로를 그대로 돌려줍니다(`local_artifact_repo.py:247`). 그래서 이 파일이 `mlruns/1/models/m-…/artifacts/`에 **직접** 생기고, 모델 폴더의 파일이 7개에서 8개가 됩니다. 불러오기만 했는데 기록 저장소에 쓰기가 일어난 것입니다. 4편 서버에서는 임시 폴더에 내려받으므로 버킷에는 이 파일이 생기지 않습니다.
3. [`forecast`, 60-61줄](predict.py#L60-L61): `feature_matrix`로 학습 때와 같은 순서의 float32 행렬을 만들어 넣고, 돌아온 **변화량**에 지금 온도(`OT_lag0`)를 더합니다.
4. [88줄](predict.py#L88): 정답과 비교해 MAE를 냅니다.

### 관찰

| champion | 출처 run | 검증 MAE (2017Q2) | 2017Q3 MAE |
|---|---|---|---|
| 버전 1 | `learning_rate=0.2` | 1.7897 | 1.7191 |
| 버전 2 | `learning_rate=0.05` | 1.7903 | **1.6568** |

- `predict.py`를 한 글자도 바꾸지 않았는데 첫 줄의 버전, run, 모델 id와 MAE가 바뀝니다.
- `list`에는 버전마다 별칭, 출처 run, mae, learning_rate, git 커밋이 나옵니다(11단계에서 이 코드를 읽습니다).
- UI: 버전 1 화면의 **Last Modified** 시각은 별칭을 붙이거나 떼도 바뀌지 않습니다. `etth1-ot` 화면의 Versions 표 Aliases 칸에서 champion이 버전 2로 옮겨 가 있습니다.

**캡처 `mlops-registry.png`**: `etth1-ot` 화면의 Versions 표를 찍습니다. 버전 두 개와, Aliases 칸에서 버전 2에 붙은 champion이 보여야 합니다. 이 화면이 레지스트리의 전부입니다. 이름 하나 아래에 후보(버전)가 쌓이고, 별칭 하나가 "지금 쓰는 것"을 가리킵니다. 노트 03 `### 실습 2`의 결과 표 다음에 있는 `<!-- 캡처 자리: ![[mlops-registry.png]] -->` 자리에 넣습니다.

### 원리

- 별칭은 `registered_model_aliases` 표의 한 행(이름, 별칭, 버전)입니다. 옮기는 것은 그 행의 버전 숫자 하나를 바꾸는 일이고, 되돌리는 것도 숫자를 다시 바꾸는 일입니다. 이것이 노트의 "서빙 코드 수정 없이"입니다.
- **언제 누가 옮겼는지는 남지 않습니다.** 표에는 지금 값만 있고 이력 표가 없습니다. Last Modified가 그대로인 것이 그 증거입니다. 운영에서 "어제 밤 9시에 어느 모델이 돌고 있었나"에 답하려면, 별칭을 옮기는 쪽(파이프라인)이 따로 로그를 남겨야 합니다.
- `predict.py`는 별칭으로 버전 번호를 **한 번** 찾고(37줄), 그 번호로 모델을 불러옵니다(50줄). `models:/etth1-ot@champion`으로 바로 불러오면, 찾는 순간과 불러오는 순간 사이에 누가 별칭을 옮겼을 때 로그에 찍힌 버전과 실제로 불러온 모델이 달라질 수 있습니다.
- 레지스트리는 `model.skops`만 관리합니다. 3번의 "변화량에 지금 온도 더하기"는 모르기 때문에, 그 일은 모델을 쓰는 쪽 코드가 맡습니다. 5편에서 ONNX로 내보낼 때 이 덧셈을 모델 그래프 안에 넣습니다.
- 검증에서 0.0006 앞선 버전 1이 다음 분기에서는 0.06 뒤졌습니다. 검증 구간 하나에서 난 작은 차이는 모델을 고르는 근거가 되지 못합니다. 여러 기간으로 번갈아 검증하는 방법(backtesting)이 필요한 이유이고, 8편의 하이퍼파라미터 탐색에서 다시 다룹니다.

## 11. 버전에서 거슬러 올라가기

> 목표: 노트의 "이 모델 어디서 나왔지"를 코드와 UI로 따라가 보고, 어디서 끊기는지 확인한다.

### 코드가 지나가는 길

`register.py list` → [`list_versions`, 112-133줄](register.py#L112-L133)

```python
versions = client.search_model_versions(f"name = '{model_name}'")  # 120: model_versions 표
run = client.get_run(version.run_id)  # 124: 버전 행의 run_id → runs 표
run.data.metrics.get("mae")  # 125: ⑦
run.data.params.get("learning_rate")  # 130: ⑤
run.data.tags.get("mlflow.source.git.commit")  # 131: ③
```

버전 → run → 파라미터, 지표, 커밋으로 표 사이를 건너갑니다. 이 사슬의 고리는 8단계 3번에서 버전 행에 함께 적은 `run_id` 하나입니다.

### 관찰

**UI**

1. Model registry → `etth1-ot` → `Version 2` → **Source Run** 링크를 누릅니다.
2. run 화면 Overview에서 Parameters(learning_rate 0.05 등), Metrics(mae 1.7903), Datasets(지문 `afa39432`), About this run의 Source(`train.py`, 브랜치, 커밋)를 봅니다. `list` 한 줄에 나온 것과 같은 길입니다.

**부숴 보기**: 커밋하지 않은 수정은 어떻게 기록될까요?

```bash
echo "# 커밋 안 한 수정" >> practice/ch03/train.py
bash practice/ch03/run.sh python practice/ch03/train.py --param learning_rate=0.06
git checkout practice/ch03/train.py   # 되돌리기
```

새 run의 Source에 찍힌 커밋은 수정 전과 같습니다. 수정이 있었다는 표시도 없습니다. 이유는 ③의 `mlflow/utils/git_utils.py:50-71`에 있습니다.

```python
repo = Repo(path, search_parent_directories=True)
if path in repo.ignored(path):
    return None
return repo.head.commit.hexsha  # HEAD 커밋만 읽는다. 작업 폴더가 깨끗한지는 보지 않는다
```

### 원리

버전 → run → 파라미터, 지표, 데이터 지문, 커밋까지는 이어집니다. 하지만 끊기는 곳이 셋 있습니다.

| 끊기는 곳 | 이유 | 메우는 방법 |
|---|---|---|
| 코드 | 커밋 해시만 남고, 커밋하지 않은 수정은 보이지 않는다(위 코드) | 학습 전에 작업 폴더가 깨끗한지 검사하거나, 파이프라인이 커밋된 코드만 돌린다(6편) |
| 데이터 | 지문은 "같은 데이터인가"만 알려 준다. 데이터가 어디 있는지는 모른다(⑤) | 데이터 버전 관리(9편) |
| 환경 | 6단계처럼 `requirements.txt`가 실제와 다를 수 있다(⑧-4) | 의존성을 직접 적거나 이미지 태그를 남긴다 |

## 12. 부숴 보기: 열 순서를 바꿔 넣으면

> 목표: 입력 형식(signature)이 무엇을 막고 무엇을 못 막는지 확인한다. 03 확인 질문 3번을 실험으로 푼다.

**예측**: 모델에 열 두 개의 순서를 바꿔 넣으면 MLflow가 막아 줄까요? float64 행렬을 넣으면? 열을 하나 빼면?

**실행**

```bash
bash practice/ch03/run.sh python practice/ch03/swap_columns.py
```

### 코드가 지나가는 길

1. [`swap_columns.py:84-86`](swap_columns.py#L84-L86): 10단계의 `predict.py`와 같은 방법으로 champion 버전을 찾아 pyfunc로 불러옵니다. [88줄](swap_columns.py#L88)이 불러온 모델의 입력 형식을 출력합니다.
2. [`swap_columns.py:96-111`](swap_columns.py#L96-L111): 같은 2017Q3 데이터를 네 가지 모양으로 넣습니다. 바꿔 넣는 두 열은 [38줄](swap_columns.py#L38)의 `HUFL`(고부하 유효 전력)과 `OT_lag0`(지금 오일 온도)입니다. 값의 범위가 겹쳐서 행렬만 봐서는 바뀐 것을 알아채기 어렵습니다.
3. `model.predict`는 예측하기 전에 MLflow의 입력 검사(`mlflow/models/utils.py:1140`의 `_enforce_schema`)를 거칩니다. 이 모델의 signature는 텐서 형식이라 `_enforce_tensor_spec`(`utils.py:692`)으로 갑니다. 여기서 검사하는 것은 **dtype**(734줄)과 **shape**(723, 730줄) 둘뿐입니다.
4. [`swap_columns.py:114-131`](swap_columns.py#L114-L131): 같은 모델을 DataFrame 입력 예시로 임시 폴더에 다시 저장합니다(118줄). 그러면 signature가 "열 이름: 타입" 목록이 되고, 검사는 `_enforce_named_col_schema`(`utils.py:970`)로 갑니다. 이 검사는 열을 **이름으로** 맞춰 넣고, 빠진 열은 이름을 대며 거부합니다(1264줄). 임시 폴더에만 저장하므로 실험 기록과 레지스트리에는 아무것도 남지 않습니다.

### 관찰

champion이 버전 2일 때 이 PC에서 본 출력입니다.

```text
입력 형식: [Tensor('float32', (-1, 15))]
1. 학습 때와 같은 순서                     MAE 1.6568
2. HUFL와 OT_lag0를 바꾼 순서              MAE 1.6859
3a. float64 행렬                           거부: dtype of input float64 does not match expected dtype float32
3b. 열 하나를 뺀 행렬(14개)                거부: Shape of input (2208, 14) does not match expected shape (-1, 15).

입력 형식: ['HUFL': float (required), 'HULL': float (required), … 'dayofweek': float (required)]
4a. 같은 순서의 DataFrame                  MAE 1.6568
4b. 순서를 바꾼 DataFrame                  MAE 1.6568
4c. OT_lag0 열이 빠진 DataFrame            거부: Model is missing inputs ['OT_lag0'].
```

3a, 3b의 거부 문구는 3번에서 본 `utils.py`의 문구와 같고, 4c는 1264줄의 문구와 같습니다.

### 원리

- 지금 모델의 입력 형식은 "float32 텐서, 열 15개"입니다. 열 이름이 없으니 MLflow가 검사할 수 있는 것은 타입과 개수뿐입니다. 순서가 바뀌면 오류 없이 **조용히 틀린 값**을 냅니다.
- 2번의 차이(+0.029)는 작아 보이지만, 10단계에서 버전 1과 2를 가른 검증 차이(0.0006)의 약 50배입니다. 그리고 아무도 모릅니다. 오류가 나는 실수보다 조용한 실수가 훨씬 위험합니다.
- 이 모델이 텐서 형식인 이유는 ⑧의 `input_example`로 numpy 행렬(`feature_matrix`)을 줬기 때문입니다. 5편에서 ONNX와 Triton으로 서빙하려고 float32 행렬로 맞췄습니다.
- 4번처럼 입력 예시를 DataFrame으로 주면 형식이 "열 이름: 타입" 목록이 됩니다. 03 확인 질문 3번의 답은 "막을 수 있다. 입력 형식을 열 이름 기반으로 남기면 된다"입니다.
- 다만 Triton 같은 서빙 엔진은 결국 텐서를 받습니다. 그때는 열 순서를 모델과 함께 저장하고(5편의 내보내기 설정에 `features` 목록이 들어가는 이유), 쓰는 쪽이 그 목록으로 행렬을 만들게 해야 합니다.

---

## 노트에 쓸 것

- run 하나에 남는 것과 그 순서: 1단계의 ①~⑩. 특히 파라미터 → 학습 → 지표 → 모델 순서가 "실패하면 무엇이 남나"를 정한다는 것(7단계)
- autolog가 더 남기는 것과 놓치는 것: 4단계 비교 표. "함수 경계 안쪽만 본다"
- 로컬 저장 위치와 형식: 메타데이터는 `mlflow.db` 하나에, 파일은 `mlruns/`에. 모델 파일은 run 폴더가 아니라 `mlruns/<실험>/models/<모델 id>/`에. 경로는 컨테이너 안 절대 경로
- MLflow 3.x의 변화: 모델이 run의 산출물에서 독립 객체(LoggedModel, `models:/m-…`)가 되었다. `log_model`의 `artifact_path` 인자가 `name`으로 바뀌었다. 모델의 지표는 run 지표 중 step 0의 복사본이다(5단계)
- run 상태의 한계(7단계), 별칭 이력이 없다는 점(10단계), 추적이 끊기는 세 곳(11단계), 입력 형식의 한계(12단계)
- 로컬 레지스트리에 별도 DB가 필요했는가: 필요 없었다. 같은 `mlflow.db`에 `registered_models`, `model_versions`, `registered_model_aliases` 표로 들어간다

## 생각해 볼 질문

1. 7단계의 RUNNING run이 1주일 뒤에도 남아 있습니다. 이것을 자동으로 정리하는 작업을 만든다면, 몇 시간이 지난 run을 "죽었다"고 판단하겠습니까? 그 기준이 틀리면 어떤 일이 생깁니까?
2. 6단계의 205줄짜리 `requirements.txt`를 그대로 두고 5편에서 서빙 이미지를 만들면 무엇이 문제입니까? 반대로 의존성을 너무 적게 적으면 어떻게 됩니까?
3. 별칭 이력이 없는 MLflow에서 "지난주 화요일 오후 3시에 champion이 몇 번 버전이었나"에 답해야 합니다. 무엇을 미리 해 뒀어야 합니까?
4. 10단계 결과를 보면, 검증 MAE가 가장 좋은 run을 자동으로 등록하고 champion으로 옮기는 파이프라인은 위험합니다. 무엇을 더 확인하게 만들겠습니까?
5. 5단계에서 모델의 곡선 지표가 step 0 값이라는 것을 봤습니다. Models 화면에서 모델끼리 비교할 지표를 제대로 남기려면 `train.py`의 어디를 어떻게 바꾸겠습니까?

## 그 밖에 보이는 경고

1단계 터미널 표에 없는 경고입니다. 실습에는 영향이 없습니다.

- `Found torch version (2.15.0+cpu) contains a local version label`: `MLFLOW_UV_AUTO_DETECT=false`로 돌릴 때(6단계) 의존성 목록을 추론하면서, `+cpu`를 떼고 적었다는 뜻입니다
- autolog run의 `` `artifact_path` is deprecated ``: MLflow의 autolog 코드가 스스로 옛 인자를 써서 나옵니다

run의 Source에 커밋이 없으면 `bash practice/ch03/run.sh git status`를 돌려 봅니다. `detected dubious ownership`이 나오면 컨테이너 안에서 저장소 파일의 소유자가 실행 사용자와 다르게 보이는 환경(예: Docker Desktop)입니다. ③의 `git_utils.py`가 예외를 만나 커밋을 비워 둡니다. 이 PC(WSL2)에서는 커밋이 남는 것을 확인했습니다.

## 4편으로 넘어가면

같은 코드를 트래킹 서버로 보내려면 4편 스택을 띄운 뒤 `run.sh` 대신 `docker compose run --rm workspace python practice/ch03/train.py`로 돌립니다. ①에서 지웠던 `MLFLOW_TRACKING_URI=http://mlflow:5000`이 그대로 살아나므로 코드는 한 줄도 바꾸지 않습니다. 그러면 ②~⑨의 MLflow 호출이 SQLite에 쓰는 대신 HTTP 요청이 되어 서버로 갑니다. 메타데이터는 PostgreSQL에, 모델 파일은 SeaweedFS에 들어갑니다. 4편 실습서는 이 HTTP 요청을 서버 로그에서 하나씩 짚는 것으로 시작합니다. [4편 실습 안내](../ch04/README.md)로 이어집니다.

## 참고 문서

- [MLflow 3 소개](https://mlflow.org/docs/latest/ml/mlflow-3/): LoggedModel과 바뀐 모델 기록 방식
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [자동 기록(autolog)](https://mlflow.org/docs/latest/ml/tracking/autolog/), [run 검색 문법](https://mlflow.org/docs/latest/ml/search/search-runs/)
- [백엔드 저장소](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/): 로컬 기본 저장 위치
- [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/), [모델 signature](https://mlflow.org/docs/latest/ml/model/signatures/), [mlflow.lightgbm API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.lightgbm.html)
