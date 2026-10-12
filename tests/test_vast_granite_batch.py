"""Batch wrapper preflight and guaranteed teardown are exercised without a rental."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("vast_granite_batch")


def test_untrusted_endpoint_is_rejected_before_any_request(batch, monkeypatch):
    monkeypatch.setattr(batch.httpx, "get", lambda *a, **kw: pytest.fail("network called"))
    with pytest.raises(ValueError, match="loopback SSH tunnel"):
        batch._preflight("https://public.example/v1")


def test_batch_closes_registered_instance_after_preflight_error(batch, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "_lease_ready", lambda id: calls.append(("lease", id)))
    monkeypatch.setattr(batch, "_deadline_ready", lambda: calls.append(("deadline",)))
    monkeypatch.setattr(batch, "_timer_ready", lambda: calls.append(("timer",)))
    monkeypatch.setattr(batch, "_preflight", lambda base: (_ for _ in ()).throw(
        ValueError("synthetic compatibility failure")))
    monkeypatch.setattr(batch.vast_lease, "check", lambda path, **kw: (
        calls.append(("close", kw["expected_id"])),
        {"status": "destroyed_and_verified"})[1])
    monkeypatch.setenv("EMBED_MODEL", "qwen3-embedding:4b")
    with pytest.raises(ValueError, match="compatibility failure"):
        batch.run(781, "http://127.0.0.1:18080/v1", pilot_rows=2, workers=1,
                  pilot_only=True)
    assert calls == [("lease", 781), ("deadline",), ("timer",), ("close", 781)]


def test_batch_never_runs_if_lease_id_not_verified(batch, monkeypatch):
    monkeypatch.setattr(batch, "_lease_ready", lambda id: (_ for _ in ()).throw(
        ValueError("missing lease")))
    monkeypatch.setattr(batch.vast_lease, "check", lambda *a, **kw: pytest.fail(
        "no matching lease: must not destroy arbitrary instance"))
    with pytest.raises(ValueError, match="missing lease"):
        batch.run(782, "http://127.0.0.1:18080/v1", pilot_rows=2, workers=1,
                  pilot_only=True)


def test_batch_pilots_then_validates_full_before_close(batch, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(batch, "_lease_ready", lambda id: calls.append("lease"))
    monkeypatch.setattr(batch, "_deadline_ready", lambda: calls.append("deadline"))
    monkeypatch.setattr(batch, "_timer_ready", lambda: calls.append("timer"))
    monkeypatch.setattr(batch, "_preflight", lambda base: calls.append("preflight"))
    def pilot(settings, env, rows, workers):
        assert env["OLLAMA_HOST"] == "http://127.0.0.1:11435"
        assert env["EMBED_MODEL"] == "qwen3-embedding:4b"
        calls.append("pilot")
    monkeypatch.setattr(batch, "_pilot", pilot)
    output = tmp_path / "verified-report.json"
    monkeypatch.setattr(batch, "_full", lambda *args: (calls.append("full"), output)[1])
    monkeypatch.setattr(batch.vast_lease, "check", lambda *args, **kwargs: (
        calls.append("close"), {"status": "destroyed_and_verified"})[1])
    result = batch.run(783, "http://127.0.0.1:18080/v1", pilot_rows=2, workers=1,
                       pilot_only=False, allow_private_remote_processing=True)
    assert result == str(output)
    assert calls == ["lease", "deadline", "timer", "preflight", "pilot", "full", "close"]


def test_expired_matching_lease_still_closes(batch, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "_lease_ready", lambda id: calls.append("identity"))
    monkeypatch.setattr(batch, "_deadline_ready", lambda: (_ for _ in ()).throw(
        ValueError("Registered lease has expired")))
    monkeypatch.setattr(batch.vast_lease, "check", lambda *args, **kwargs: (
        calls.append(("close", kwargs["expected_id"])),
        {"status": "destroyed_and_verified"})[1])
    with pytest.raises(ValueError, match="expired"):
        batch.run(784, "http://127.0.0.1:18080/v1", pilot_rows=2, workers=1,
                  pilot_only=True)
    assert calls == ["identity", ("close", 784)]


def test_private_pilot_needs_explicit_gate_even_after_synthetic_probe(batch, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "_lease_ready", lambda id: None)
    monkeypatch.setattr(batch, "_deadline_ready", lambda: None)
    monkeypatch.setattr(batch, "_timer_ready", lambda: None)
    monkeypatch.setattr(batch, "_preflight", lambda base: calls.append("synthetic"))
    monkeypatch.setattr(batch, "_pilot", lambda *args: pytest.fail("private pilot sent"))
    monkeypatch.setattr(batch.vast_lease, "check", lambda *args, **kwargs: (
        calls.append("closed"), {"status": "destroyed_and_verified"})[1])
    with pytest.raises(PermissionError, match="explicit approval"):
        batch.run(785, "http://127.0.0.1:18080/v1", pilot_rows=2, workers=1,
                  pilot_only=True)
    assert calls == ["synthetic", "closed"]


def test_pilot_retry_checks_latest_run_only(batch, monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(batch, "_run", lambda *args: None)
    rows = pd.DataFrame([
        {"run_id": run, "row_index": row, "label": label, "parsed": "yes"}
        for run in (0, 1) for row in (0, 1) for label in ("Initiative", "Optimism")
    ])
    monkeypatch.setattr(batch, "read_raw_results", lambda *a: rows)
    settings = SimpleNamespace(artifacts_dir=tmp_path / "granite",
                               run_prefix="f25_core_", active_labels=("Initiative", "Optimism"))
    batch._pilot(settings, {}, 2, 1)


def test_full_resume_skips_verified_completed_sheet(batch, monkeypatch, tmp_path):
    from types import SimpleNamespace
    out = tmp_path / "granite"
    out.mkdir()
    (out / "f25_core_raw_results.xlsx").touch()
    settings = SimpleNamespace(artifacts_dir=out, run_prefix="f25_core_")
    checked = []
    monkeypatch.setitem(sys.modules, "evaluate_reconciled", SimpleNamespace(
        score_one=lambda s, term, team, meet, **kw: checked.append((term, team))))
    monkeypatch.setattr(batch, "read_raw_results", lambda path, sheet:
                        pd.DataFrame([{"run_id": 0}]) if "S26" in sheet and "SP26" not in sheet
                        else pd.DataFrame())
    commands = []

    def run(cmd, env):
        commands.append(cmd)
        if "scripts/evaluate_reconciled.py" in cmd:
            (out / "reconciled-evaluation.json").write_text(
                '{"sheets":[{},{}],"totals":{"rows":2,"jobs":2,"tp":1,"fp":0,'
                '"fn":0,"tn":1,"abstained_positive":0,"abstained_negative":0}}')

    monkeypatch.setattr(batch, "_run", run)
    assert batch._full(settings, {}, 1) == out / "reconciled-evaluation.json"
    assert checked == [("S26", 1)]
    infers = [cmd for cmd in commands if "infer" in cmd]
    assert len(infers) == 1 and "SP26" in infers[0]
