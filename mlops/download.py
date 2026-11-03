"""원본 데이터 내려받기. 출처는 커밋이나 리비전으로 고정해서 어느 PC에서든 같은 파일을 받는다."""

import hashlib
import logging
import posixpath
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download

from mlops.config import RAW_CACHE_DIR

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 120
CHUNK_BYTES = 1 << 20
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url: str, sha256: str, cache_dir: Path = RAW_CACHE_DIR) -> Path:
    """URL의 파일을 캐시에 받아 sha256을 확인한다. 캐시에 맞는 파일이 있으면 다시 받지 않는다."""
    if not SHA256_HEX.fullmatch(sha256):
        raise ValueError(f"sha256은 소문자 16진수 64자리다: {sha256!r}")
    # Path(...).name은 끝의 /를 무시해서 '/data/'를 'data'로 읽는다. URL 경로 그대로 본다
    name = posixpath.basename(urlparse(url).path)
    if not name:
        raise ValueError(f"URL 끝에 파일 이름이 없다: {url}")
    dest = Path(cache_dir) / sha256 / name
    if dest.is_file() and _sha256(dest) == sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("내려받는 중: %s", url)
    # 임시 파일 이름을 따로 받아서, 두 컨테이너가 같은 파일을 동시에 받아도 서로 덮어쓰지 않게 한다
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as f:
        partial = Path(f.name)
    try:
        with (
            urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response,
            partial.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
        actual = _sha256(partial)
        if actual != sha256:
            raise ValueError(f"{url}: sha256이 다르다 (기대 {sha256}, 실제 {actual})")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def fetch_hf_dataset_file(repo_id: str, filename: str, revision: str) -> Path:
    """Hugging Face 데이터셋 파일을 리비전(커밋 해시)으로 고정해 받는다. HF_HOME에 캐시된다."""
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision))
