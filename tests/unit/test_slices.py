import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlops.slices import read_manifest, read_slice, read_slices, write_slices


def test_written_slices_read_back_in_order(tmp_path):
    slices = {"a": pd.DataFrame({"x": [1, 2]}), "b": pd.DataFrame({"x": [3]})}

    write_slices(tmp_path, "demo", slices, source={"url": "https://example.com/data.csv"})

    assert_frame_equal(read_slice(tmp_path, "demo", "b"), slices["b"])
    assert read_slices(tmp_path, "demo", ["a", "b"])["x"].tolist() == [1, 2, 3]


def test_manifest_records_source_and_row_counts(tmp_path):
    slices = {"a": pd.DataFrame({"x": [1, 2]}), "b": pd.DataFrame({"x": [3]})}

    write_slices(tmp_path, "demo", slices, source={"url": "https://example.com/data.csv"})

    manifest = read_manifest(tmp_path, "demo")
    assert manifest["source"] == {"url": "https://example.com/data.csv"}
    assert manifest["slices"] == [{"name": "a", "rows": 2}, {"name": "b", "rows": 1}]


def test_missing_slice_tells_how_to_prepare_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m mlops prepare demo"):
        read_slice(tmp_path, "demo", "a")


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "a\n"])
def test_slice_names_cannot_leave_the_task_directory(tmp_path, name):
    with pytest.raises(ValueError, match="슬라이스 이름"):
        write_slices(tmp_path, "demo", {name: pd.DataFrame({"x": [1]})}, source={})
