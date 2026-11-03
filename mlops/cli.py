"""명령줄 진입점.

python -m mlops prepare <task>
python -m mlops train <task> [--param key=value ...] [--max-eval-rows N]
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mlops.config import ARTIFACTS_DIR, DATA_DIR
from mlops.runner import ExportMismatchError, parse_params, run_prepare, run_training
from mlops.tasks import TASK_NAMES, get_task

logger = logging.getLogger("mlops")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mlops", description="MLOps 입문 실습 과제")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="원본을 받아 시간 순서 슬라이스로 나눈다")
    prepare.add_argument("task", choices=TASK_NAMES)
    prepare.add_argument("--data-dir", type=Path, default=DATA_DIR)

    train = commands.add_parser("train", help="학습하고 슬라이스별로 평가한 뒤 내보낸다")
    train.add_argument("task", choices=TASK_NAMES)
    train.add_argument("--data-dir", type=Path, default=DATA_DIR)
    train.add_argument("--out", type=Path, help=f"결과 폴더 (기본: {ARTIFACTS_DIR}/<task>)")
    train.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="기본 파라미터를 바꾼다. 여러 번 줄 수 있다",
    )
    train.add_argument(
        "--max-eval-rows", type=int, help="슬라이스마다 평가할 최대 행 수 (기본: 과제마다 다름)"
    )
    train.add_argument(
        "--train-slices", type=_names, help="학습 슬라이스, 쉼표로 구분 (기본: 과제마다 다름)"
    )
    train.add_argument("--valid-slices", type=_names, help="검증 슬라이스, 쉼표로 구분")
    return parser


def _names(text: str) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()]


def _role(report: dict[str, Any], name: str) -> str:
    if name in report["train_slices"]:
        return "train"
    return "valid" if name in report["valid_slices"] else "future"


def _format(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _log_summary(report: dict[str, Any]) -> None:
    for name, metrics in report["metrics"].items():
        values = " ".join(f"{k}={_format(v)}" for k, v in metrics.items())
        logger.info("%-8s %-6s %s", name, _role(report, name), values)
    logger.info("내보내기 확인: %s", report["export_check"])
    logger.info("소요 시간(초): %s, 최대 메모리: %s MB", report["seconds"], report["peak_rss_mb"])


def main(argv: Sequence[str] | None = None) -> int:
    # 라이브러리(HTTP 클라이언트 등)는 경고만, 이 패키지는 진행 상황까지 보여 준다
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)
    args = build_parser().parse_args(argv)
    task = get_task(args.task)
    try:
        if args.command == "prepare":
            manifest = run_prepare(task, args.data_dir)
            logger.info("슬라이스를 저장했다: %s", manifest.parent)
            return 0
        params = parse_params(task.default_params, args.param)
        out_dir = args.out or ARTIFACTS_DIR / task.name
        report = run_training(
            task,
            args.data_dir,
            out_dir,
            params,
            args.max_eval_rows,
            train_slices=args.train_slices,
            valid_slices=args.valid_slices,
        )
    # OSError: 내려받기 실패(네트워크), 디스크 오류
    except (OSError, ValueError, ExportMismatchError) as error:
        logger.error("%s", error)
        return 1
    _log_summary(report)
    logger.info("결과: %s", out_dir)
    return 0
