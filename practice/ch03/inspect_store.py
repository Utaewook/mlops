"""로컬 MLflow 저장소(mlflow.db와 mlruns/)에 무엇이 어떤 모양으로 들어갔는지 보여 준다.

    bash practice/ch03/run.sh python practice/ch03/inspect_store.py

MLflow API를 거치지 않고 SQLite 파일과 폴더를 직접 읽는다. UI와 API가 보여 주는 것이 실제로는
"표 몇 개와 파일 몇 개"라는 것을 확인하려는 용도다. 읽기만 하고 아무것도 고치지 않는다.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import mlflow

# 3편에서 볼 만한 표와 설명. 나머지 표(트레이스, 프롬프트, 평가 등 GenAI 기능용)는 이름과 행 수만 보여 준다
TABLE_NOTES = {
    "experiments": "실험. 이름과 산출물 위치(artifact_location)",
    "runs": "run. 상태(RUNNING/FINISHED/FAILED), 시작·끝 시각, 산출물 위치",
    "params": "파라미터. (run_uuid, key, value) — 값은 문자열",
    "metrics": "지표의 모든 기록. (run_uuid, key, value, step, timestamp)",
    "latest_metrics": "지표마다 마지막 값만. run 목록 화면이 빨리 뜨게 따로 둔다",
    "tags": "run 태그. mlflow.source.git.commit 같은 자동 태그 포함",
    "datasets": "log_input으로 남긴 데이터 요약(이름, 지문, 열 구조, 행 수)",
    "inputs": "어느 run이 어느 데이터를 썼는가(run ↔ dataset 연결)",
    "logged_models": "MLflow 3.x의 모델(m-...). 어느 run에서 나왔는지, 파일 위치",
    "logged_model_metrics": "모델에 연결된 지표",
    "registered_models": "레지스트리의 등록 모델 이름",
    "model_versions": "등록 모델의 버전. 모델 주소(source)와 run_id",
    "registered_model_aliases": "별칭 → 버전",
    "input_tags": "데이터 입력의 용도(training, validation)",
    "logged_model_params": "모델에 연결된 파라미터(만든 run에서 복사)",
    "logged_model_tags": "모델 태그",
    "alembic_version": "DB 구조의 버전. MLflow를 올리면 이 값을 보고 표 구조를 고친다",
}
TREE_MAX_DEPTH = 6
TREE_MAX_ENTRIES = 12


def sqlite_path() -> Path:
    uri = mlflow.get_tracking_uri()
    if not uri.startswith("sqlite:"):
        raise SystemExit(f"트래킹 주소가 SQLite가 아니다: {uri}. run.sh로 실행한다")
    # SQLAlchemy 주소 규칙: sqlite:/// 뒤가 파일 경로다. 슬래시가 4개면 절대 경로
    # (sqlite:////workspace/mlflow.db → /workspace/mlflow.db)
    return Path(uri.removeprefix("sqlite:///"))


def show_tables(db: Path) -> None:
    print(f"== {db} ({db.stat().st_size / 1024:.0f} KiB)")
    # mode=ro: 읽기 전용으로 연다. UI나 학습이 같은 파일을 쓰고 있어도 건드리지 않는다.
    # sqlite3 연결의 with는 트랜잭션만 끝내고 연결을 닫지 않으므로 closing으로 감싼다
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        counts = {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in names
        }
    for name in [*TABLE_NOTES, *sorted(set(names) - set(TABLE_NOTES))]:
        if name not in counts:
            continue
        note = TABLE_NOTES.get(name, "")
        if note or counts[name]:
            print(f"  {name:<26} {counts[name]:>6}행  {note}")
    hidden = [n for n in names if n not in TABLE_NOTES and not counts[n]]
    print(f"  (비어 있는 기타 표 {len(hidden)}개 생략)")


def show_tree(root: Path, depth: int = 0) -> None:
    if depth > TREE_MAX_DEPTH:
        return
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    for entry in entries[:TREE_MAX_ENTRIES]:
        indent = "  " * (depth + 1)
        if entry.is_dir():
            print(f"{indent}{entry.name}/")
            show_tree(entry, depth + 1)
        else:
            print(f"{indent}{entry.name} ({entry.stat().st_size / 1024:.1f} KiB)")
    if len(entries) > TREE_MAX_ENTRIES:
        print(f"{'  ' * (depth + 1)}… {len(entries) - TREE_MAX_ENTRIES}개 더")


def main() -> None:
    db = sqlite_path()
    if not db.exists():
        raise SystemExit(f"{db}가 없다. train.py를 먼저 돌린다")
    show_tables(db)

    # 실험을 만들 때 정해진 산출물 위치. run.sh는 저장소 루트에서 돌므로 ./mlruns다
    artifacts = Path("mlruns")
    print(f"\n== {artifacts.resolve()}/")
    if artifacts.exists():
        show_tree(artifacts)
    else:
        print("  (아직 없다)")


if __name__ == "__main__":
    main()
