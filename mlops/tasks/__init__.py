"""과제 등록부. 과제 모듈은 필요할 때만 import한다(Time Series만 돌릴 때 torch를 올리지 않도록)."""

import importlib

from mlops.tasks.base import Task

TASK_NAMES = ("timeseries", "vision", "llm")


def get_task(name: str) -> Task:
    if name not in TASK_NAMES:
        raise ValueError(f"알 수 없는 과제: {name!r}. 다음 중 하나: {', '.join(TASK_NAMES)}")
    return importlib.import_module(f"mlops.tasks.{name}").TASK
