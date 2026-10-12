"""No-network lease watchdog tests: only the registered Vast instance may be destroyed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/vast_lease.py"
    spec = importlib.util.spec_from_file_location("vast_lease", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expired_lease_destroys_only_own_instance_and_verifies(tmp_path, monkeypatch):
    module = _load()
    path = tmp_path / "lease.json"
    calls = []
    state = {"active": True}

    def cli(*args):
        calls.append(args)
        if args[:2] == ("show", "instances-v1"):
            rows = [{"id": 87}, {"id": 88}] if state["active"] else [{"id": 88}]
            return json.dumps({"success": True, "instances": rows,
                               "instances_found": len(rows), "total_instances": len(rows),
                               "next_token": None})
        assert args == ("destroy", "instance", "87", "-y")
        state["active"] = False
        return "ok"

    monkeypatch.setattr(module, "_cli", cli)
    lease = module.register(path, 87, 2, now=100)
    assert lease["destroy_at_epoch"] == 7300
    assert path.stat().st_mode & 0o777 == 0o600
    assert module.check(path, now=7299)["status"] == "leased"
    assert calls == []
    assert module.check(path, now=7300)["status"] == "destroyed_and_verified"
    assert not path.exists()
    assert calls == [("show", "instances-v1", "--raw"),
                     ("destroy", "instance", "87", "-y"),
                     ("show", "instances-v1", "--raw")]
    assert module.check(path, now=7400)["status"] == "no_lease"


def test_close_on_success_or_failure_destroys_immediately(tmp_path, monkeypatch):
    module = _load()
    path = tmp_path / "lease.json"
    module.register(path, 123, 4, now=100)
    state = {"active": True}

    def cli(*args):
        if args[:2] == ("show", "instances-v1"):
            rows = [{"id": 123}] if state["active"] else []
            return json.dumps({"success": True, "instances": rows,
                               "instances_found": len(rows), "total_instances": len(rows),
                               "next_token": None})
        assert args == ("destroy", "instance", "123", "-y")
        state["active"] = False
        return ""

    monkeypatch.setattr(module, "_cli", cli)
    assert module.check(path, now=200, close=True)["status"] == "destroyed_and_verified"


def test_provider_failure_retains_lease_for_next_timer_tick(tmp_path, monkeypatch):
    module = _load()
    path = tmp_path / "lease.json"
    module.register(path, 999, 1, now=100)
    monkeypatch.setattr(module, "_cli", lambda *args: "[]" if args[0] == "show"
                        else (_ for _ in ()).throw(RuntimeError("destroy blocked")))
    # If provider says instance is already absent, clearing the lease is safe.
    assert module.check(path, now=4000)["status"] == "destroyed_and_verified"
    module.register(path, 999, 1, now=100)
    monkeypatch.setattr(module, "_cli", lambda *args: '[{"id":999}]' if args[0] == "show"
                        else (_ for _ in ()).throw(RuntimeError("destroy blocked")))
    with pytest.raises(RuntimeError, match="blocked"):
        module.check(path, now=4000)
    assert path.exists()


def test_duplicate_or_invalid_lease_is_rejected(tmp_path):
    module = _load()
    path = tmp_path / "lease.json"
    with pytest.raises(ValueError):
        module.register(path, 0, 1)
    with pytest.raises(ValueError):
        module.register(path, 1, 25)
    module.register(path, 1, 1)
    with pytest.raises(FileExistsError):
        module.register(path, 2, 1)
    with pytest.raises(ValueError, match="does not match"):
        module.check(path, now=99999, close=True, expected_id=2)
    assert path.exists()


def test_paginated_provider_list_cannot_fake_absence(tmp_path, monkeypatch):
    module = _load()
    path = tmp_path / "lease.json"
    module.register(path, 91, 1, now=0)
    monkeypatch.setattr(module, "_cli", lambda *args: json.dumps({
        "success": True, "instances": [{"id": 90}],
        "instances_found": 1, "total_instances": 2, "next_token": None}))
    with pytest.raises(RuntimeError, match="incomplete"):
        module.check(path, now=4000)
    assert path.exists()


def test_target_on_second_page_is_destroyed_not_orphaned(tmp_path, monkeypatch):
    module = _load()
    path = tmp_path / "lease.json"
    module.register(path, 200, 1, now=0)
    state = {"active": True}
    calls = []

    def cli(*args):
        calls.append(args)
        if args[0] == "destroy":
            assert args == ("destroy", "instance", "200", "-y")
            state["active"] = False
            return ""
        if "--next-token" in args:
            rows = [{"id": 200}] if state["active"] else []
            return json.dumps({"success": True, "instances": rows,
                               "instances_found": len(rows), "total_instances": 1 + len(rows),
                               "next_token": None})
        return json.dumps({"success": True, "instances": [{"id": 10}],
                           "instances_found": 1, "total_instances": 2 if state["active"] else 1,
                           "next_token": "page-two" if state["active"] else None})

    monkeypatch.setattr(module, "_cli", cli)
    assert module.check(path, now=4000)["status"] == "destroyed_and_verified"
    assert ("show", "instances-v1", "--raw", "--next-token", "page-two") in calls
    assert ("destroy", "instance", "200", "-y") in calls
    assert not path.exists()


def test_backup_watchdog_runs_provider_cli_locally(monkeypatch, tmp_path):
    monkeypatch.setenv("VAST_LEASE_CLI_MODE", "local")
    monkeypatch.setenv("VAST_LEASE_PATH", str(tmp_path / "remote-lease.json"))
    module = _load()
    assert module.LEASE == tmp_path / "remote-lease.json"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        from subprocess import CompletedProcess
        return CompletedProcess(command, 0, '{"success":true,"instances":[],"total_instances":0}', "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._active(987) is False
    assert commands == [[module.CLI, "show", "instances-v1", "--raw"]]
