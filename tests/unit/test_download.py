import hashlib

import pytest

from mlops.download import fetch_url

CONTENT = b"date,OT\n2016-07-01 00:00:00,30.5\n"
CONTENT_SHA256 = hashlib.sha256(CONTENT).hexdigest()


def test_fetch_url_downloads_once_and_then_uses_cache(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(CONTENT)
    cache_dir = tmp_path / "cache"

    first = fetch_url(source.as_uri(), CONTENT_SHA256, cache_dir=cache_dir)
    source.unlink()
    second = fetch_url(source.as_uri(), CONTENT_SHA256, cache_dir=cache_dir)

    assert first == second
    assert second.read_bytes() == CONTENT


def test_fetch_url_rejects_checksum_mismatch_and_keeps_nothing(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(CONTENT)
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match="sha256"):
        fetch_url(source.as_uri(), "0" * 64, cache_dir=cache_dir)

    assert [p for p in cache_dir.rglob("*") if p.is_file()] == []


def test_fetch_url_rejects_malformed_checksum(tmp_path):
    with pytest.raises(ValueError, match="sha256"):
        fetch_url("https://example.com/data.csv", "../../etc", cache_dir=tmp_path)


def test_fetch_url_needs_a_file_name_in_the_url(tmp_path):
    with pytest.raises(ValueError, match="파일 이름"):
        fetch_url("https://example.com/data/", CONTENT_SHA256, cache_dir=tmp_path)
