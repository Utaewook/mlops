"""경로 설정. 컨테이너에서는 Dockerfile과 compose가 환경 변수를 정해 준다."""

import os
from pathlib import Path

# 과제별 슬라이스(parquet). 저장소의 data/는 git에 올라가지 않는다
DATA_DIR = Path(os.environ.get("MLOPS_DATA_DIR", "data"))

# 학습 결과(리포트, 내보낸 모델)
ARTIFACTS_DIR = Path(os.environ.get("MLOPS_ARTIFACTS_DIR", "artifacts"))

# URL로 받은 원본 파일. Hugging Face 파일은 HF_HOME 캐시를 따로 쓴다
RAW_CACHE_DIR = Path(os.environ.get("MLOPS_CACHE_DIR", Path.home() / ".cache" / "mlops"))
