# 실습 코드(workspace), 테스트, Dagster(webserver, daemon, code)가 같이 쓰는 Python 이미지.
# 의존성은 저장소의 uv.lock으로 고정해서 어느 PC에서 빌드해도 같은 버전이 깔린다.
FROM python:3.12.16-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

# LightGBM이 쓰는 OpenMP 런타임. slim 이미지에는 없다
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/dagster_home

WORKDIR /workspace

# 의존성만 먼저 설치해서, 코드가 바뀌어도 이 층은 캐시를 쓴다. dev 그룹(pytest, ruff)도 포함.
# --locked: pyproject.toml을 고치고 uv lock을 안 돌렸으면 빌드를 실패시킨다
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# 내려받은 데이터 원본, 사전학습 가중치, Hugging Face 모델을 둘 캐시. compose가 model-cache 볼륨을 붙이고,
# 볼륨은 처음 붙을 때 이 폴더의 권한을 물려받는다. workspace는 호스트 사용자 ID로 돌기 때문에 누구나 쓸 수 있게 둔다
ENV HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    MLOPS_CACHE_DIR=/cache/mlops
RUN mkdir -p /cache && chmod 1777 /cache

# glibc malloc이 놓아 준 텐서 메모리를 OS에 돌려주게 한다. 기본값에서는 해제한 메모리가 프로세스 힙에 남아
# LLM(Qwen3-0.6B) 학습 중 RSS가 약 2GB 더 크고, 학습한 모델을 놓고 내보낸 모델을 다시 올릴 때 5GB 상한을 넘었다.
# 1MB 이상 할당을 mmap으로 받고 아레나 수를 줄이면 학습 중 RSS 4.8GB → 2.8GB, 최대 4.9GB → 4.1GB였다
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=1048576

COPY infra/dagster/dagster.yaml infra/dagster/workspace.yaml ${DAGSTER_HOME}/
