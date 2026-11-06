"""로컬 MLflow 저장소(mlflow.db와 mlruns/)에 무엇이 어떤 모양으로 들어갔는지 보여 준다.

    bash practice/ch03/run.sh python practice/ch03/inspect_store.py
    bash practice/ch03/run.sh python practice/ch03/inspect_store.py --run-name manual-lr0.05-leaves31

MLflow API를 거치지 않고 SQLite 파일과 폴더를 직접 읽는다. UI와 API가 보여 주는 것이 실제로는
"표 몇 개와 파일 몇 개"라는 것을 확인하려는 용도다. 읽기만 하고 아무것도 고치지 않는다.
"""

import argparse
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
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

# run마다 각 표에 몇 행이 있는지. 삭제 표시된 run(lifecycle_stage = 'deleted')은 뺀다
RUN_COUNTS_SQL = """
    SELECT r.name, r.status,
        (SELECT COUNT(*) FROM params p WHERE p.run_uuid = r.run_uuid),
        (SELECT COUNT(*) FROM metrics m WHERE m.run_uuid = r.run_uuid),
        (SELECT COUNT(*) FROM latest_metrics l WHERE l.run_uuid = r.run_uuid),
        (SELECT COUNT(*) FROM tags t WHERE t.run_uuid = r.run_uuid)
    FROM runs r
    WHERE r.lifecycle_stage = 'active'
    ORDER BY r.start_time
"""
# 이름이 같은 run이 여럿이면 가장 최근 것을 고른다. 이름을 주지 않으면 가장 최근 run
FIND_RUN_SQL = """
    SELECT run_uuid, name, status FROM runs
    WHERE lifecycle_stage = 'active' AND (? IS NULL OR name = ?)
    ORDER BY start_time DESC
    LIMIT 1
"""


@dataclass(frozen=True)
class RunCounts:
    name: str
    status: str
    params: int
    metrics: int
    latest_metrics: int
    tags: int


def connect_read_only(db: Path) -> sqlite3.Connection:
    # mode=ro: 읽기 전용으로 연다. UI나 학습이 같은 파일을 쓰고 있어도 건드리지 않는다
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def sqlite_path() -> Path:
    uri = mlflow.get_tracking_uri()
    if not uri.startswith("sqlite:"):
        raise SystemExit(f"트래킹 주소가 SQLite가 아니다: {uri}. run.sh로 실행한다")
    # SQLAlchemy 주소 규칙: sqlite:/// 뒤가 파일 경로다. 슬래시가 4개면 절대 경로
    # (sqlite:////workspace/mlflow.db → /workspace/mlflow.db)
    return Path(uri.removeprefix("sqlite:///"))


def show_tables(db: Path) -> None:
    print(f"== {db} ({db.stat().st_size / 1024:.0f} KiB)")
    # sqlite3 연결의 with는 트랜잭션만 끝내고 연결을 닫지 않으므로 closing으로 감싼다
    with closing(connect_read_only(db)) as conn:
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


def run_counts(db: Path) -> list[RunCounts]:
    """run마다 params, metrics, latest_metrics, tags 표에 몇 행씩 있는지 센다."""
    with closing(connect_read_only(db)) as conn:
        return [RunCounts(*row) for row in conn.execute(RUN_COUNTS_SQL)]


def show_run_counts(db: Path) -> None:
    """run별 행 수. 직접 기록과 autolog의 차이, 곡선 점과 마지막 값의 차이가 숫자로 보인다.

    metrics 표에는 같은 지표라도 step마다 한 행씩(곡선 점 전부) 들어가고, latest_metrics 표에는
    지표마다 마지막 값 한 행만 들어간다. UI의 run 목록은 latest_metrics를, 차트는 metrics를 읽는다.
    """
    print("\n== run별 행 수 (metrics: 지표 기록 전부, latest: 지표마다 마지막 값 하나)")
    print(f"  {'run 이름':<30} {'상태':<7} {'params':>6} {'metrics':>8} {'latest':>7} {'tags':>5}")
    for row in run_counts(db):
        print(
            f"  {row.name:<32} {row.status:<9} {row.params:>6} {row.metrics:>8}"
            f" {row.latest_metrics:>7} {row.tags:>5}"
        )


def show_run_tags(db: Path, run_name: str | None) -> None:
    """run 하나의 태그를 표에서 직접 읽어, UI가 감추거나 바꿔 보여 주는 원래 값을 보여 준다.

    UI의 Tags에는 직접 붙인 것만 보인다. MLflow가 스스로 붙인 mlflow.* 태그(커밋, 실행 파일,
    사용자 등)는 About this run의 Source, Created by 칸으로 모양을 바꿔 나온다.
    """
    with closing(connect_read_only(db)) as conn:
        row = conn.execute(FIND_RUN_SQL, (run_name, run_name)).fetchone()
        if row is None:
            if run_name is not None:
                raise SystemExit(f"없는 run 이름: {run_name!r}. 위 목록의 이름을 쓴다")
            print("\n== run: (아직 없다)")
            return
        run_id, name, status = row
        tags = conn.execute(
            "SELECT key, value FROM tags WHERE run_uuid = ? ORDER BY key", (run_id,)
        ).fetchall()
    print(f"\n== run {name} ({status}, {run_id})의 태그")
    for key, value in tags:
        print(f"  {key:<26} {value}")


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


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="로컬 MLflow 저장소의 표와 파일을 직접 본다")
    parser.add_argument("--run-name", help="태그를 볼 run 이름. 없으면 가장 최근 run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    db = sqlite_path()
    if not db.exists():
        raise SystemExit(f"{db}가 없다. train.py를 먼저 돌린다")
    show_tables(db)
    show_run_counts(db)
    show_run_tags(db, args.run_name)

    # 실험을 만들 때 정해진 산출물 위치. run.sh는 저장소 루트에서 돌므로 ./mlruns다
    artifacts = Path("mlruns")
    print(f"\n== {artifacts.resolve()}/")
    if artifacts.exists():
        show_tree(artifacts)
    else:
        print("  (아직 없다)")


if __name__ == "__main__":
    main()
