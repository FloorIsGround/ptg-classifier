"""An interrupted Excel export must not destroy earlier completed runs."""

import hashlib

import pandas as pd
import pytest

from ptg.storage.raw_results import RawResult, read_raw_results, write_raw_results


def test_failed_export_preserves_prior_workbook(tmp_path, monkeypatch):
    path = tmp_path / "raw.xlsx"
    rows = [RawResult(0, "00:01", "A", "synthetic", "Label", "yes", "yes")]
    write_raw_results(str(path), "Team 1 Time 1 (S26) Run 0", rows)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def abort(*args, **kwargs):
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(pd.DataFrame, "to_excel", abort)
    with pytest.raises(OSError, match="synthetic interrupted write"):
        write_raw_results(str(path), "Team 1 Time 1 (S26) Run 1", rows)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob(".raw.*.xlsx"))
    assert len(read_raw_results(str(path), "Team 1 Time 1 (S26)")) == 1
