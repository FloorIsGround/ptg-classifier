"""Run a local Granite pilot/full batch against an approved SSH-tunneled vLLM.

Good: after the user approves remote transcript processing and a budget, create
and register a single Vast instance, establish an SSH tunnel, then use this
wrapper. Bad: run on an unregistered rental or send prompts to a public URL.
Inputs, embedding/index, checkpoints and outputs remain on this workstation;
transcript prompts still transit through the rented GPU and require approval.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ptg.classify.binary_classifier import BinaryClassificationJob, classify_binary_jobs
from ptg.config import load_settings
from ptg.domain.models import CodebookEntry
from ptg.storage.raw_results import read_raw_results
import vast_lease

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config/study-21-granite-4.2.yaml"
LEASE = ROOT / "artifacts/vast-granite-lease.json"


def _lease_ready(instance_id: int) -> None:
    lease = json.loads(LEASE.read_text(encoding="utf-8"))
    if lease["instance_id"] != instance_id:
        raise ValueError("Registered lease is missing or mismatched")


def _deadline_ready() -> None:
    lease = json.loads(LEASE.read_text(encoding="utf-8"))
    if lease["destroy_at_epoch"] <= time.time():
        raise ValueError("Registered lease has expired")


def _timer_ready() -> None:
    status = subprocess.run(["systemctl", "--user", "is-active", "vast-granite-watchdog.timer"],
                            capture_output=True, text=True, check=False)
    if status.returncode or status.stdout.strip() != "active":
        raise RuntimeError("Independent Vast lease timer is not active")


def _run(args: list[str], env: dict[str, str]) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def _preflight(base: str) -> None:
    endpoint = urlsplit(base)
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or not endpoint.path.endswith("/v1"):
        raise ValueError("LM endpoint must be a loopback SSH tunnel ending in /v1")
    response = httpx.get(base.rstrip("/") + "/models", timeout=10)
    response.raise_for_status()
    names = {item["id"] for item in response.json()["data"]}
    if "granite-4.2-8b" not in names:
        raise ValueError("Tunnel is not serving granite-4.2-8b")
    entry = CodebookEntry("synthetic", "Initiative", "Synthetic", "Starting an action")
    job = BinaryClassificationJob(0, entry, [], "Let us start the meeting.", [])
    result = asyncio.run(classify_binary_jobs(
        jobs=[job], api_base=base, model="granite-4.2-8b", max_retries=0,
        temperature=1.0, top_p=0.95, max_tokens=64,
        chat_template_kwargs={"enable_thinking": False}, seed=42, timeout_s=60,
    ))
    if result[(0, "Initiative")][0] not in ("yes", "no"):
        raise ValueError("Granite synthetic yes/no probe did not parse; no private requests sent")


def _pilot(settings, env: dict[str, str], rows: int, workers: int) -> None:
    base = ["uv", "run", "--frozen", "ptg", "--config", CONFIG]
    _run(base + ["infer", "--semester", "S26", "--team", "1", "--meet", "1",
                 "--lm-workers", str(workers), "--pilot-row-limit", str(rows)], env)
    _run(base + ["export", "--semester", "S26", "--team", "1", "--meet", "1",
                 "--pilot", str(rows)], env)
    result_dir = settings.artifacts_dir.with_name(f"{settings.artifacts_dir.name}_pilot{rows}")
    raw = read_raw_results(str(result_dir / f"{settings.run_prefix}raw_results.xlsx"),
                           "Team 1 Time 1 (S26)")
    if raw.empty:
        raise RuntimeError("Pilot has no results")
    run_id = int(raw.run_id.max())
    raw = raw.loc[raw.run_id == run_id]
    selected = set(raw.row_index.astype(int))
    pairs = list(zip(raw.row_index.astype(int), raw.label.astype(str)))
    expected = {(idx, label) for idx in selected for label in settings.active_labels}
    if len(selected) != rows or len(pairs) != len(expected) or set(pairs) != expected:
        raise RuntimeError("Pilot lacks exact row-label coverage; aborting full batch")
    if not raw.parsed.astype(str).str.casefold().isin(("yes", "no")).all():
        raise RuntimeError("Pilot has invalid/failed responses; aborting full batch")


def _full(settings, env: dict[str, str], workers: int) -> Path:
    from evaluate_reconciled import score_one

    base = ["uv", "run", "--frozen", "ptg", "--config", CONFIG]
    for term, team in (("S26", 1), ("SP26", 2)):
        name = f"Team {team} Time 1 ({term})"
        raw_path = settings.artifacts_dir / f"{settings.run_prefix}raw_results.xlsx"
        previous = read_raw_results(str(raw_path), name) if raw_path.exists() else None
        if previous is not None and not previous.empty:
            # Completed sheets must match current workbook/index/model/prompt
            # before skipping; incompatible artifacts require a fresh namespace.
            score_one(settings, term, team, 1, verify_ledger=True, lm_timeout=120.0)
        else:
            # A partial SQLite checkpoint without an exported raw sheet resumes.
            _run(base + ["infer", "--semester", term, "--team", str(team),
                         "--meet", "1", "--lm-workers", str(workers)], env)
        _run(base + ["export", "--semester", term, "--team", str(team), "--meet", "1"], env)
    output = settings.artifacts_dir / "reconciled-evaluation.json"
    _run(["uv", "run", "--frozen", "python", "scripts/evaluate_reconciled.py",
          "--config", CONFIG, "--sheets", "S26:1:1,SP26:2:1",
          "--affirm-reviewed-blanks", "--out", str(output)], env)
    report = json.loads(output.read_text(encoding="utf-8"))
    totals = report["totals"]
    if len(report["sheets"]) != 2 or totals["rows"] < 1 or totals["jobs"] != (
        sum(totals[key] for key in ("tp", "fp", "fn", "tn", "abstained_positive", "abstained_negative"))
    ):
        raise RuntimeError("Local Granite evaluation incomplete; preserving checkpoint")
    return output


def run(instance_id: int, base: str, *, pilot_rows: int, workers: int,
        pilot_only: bool, allow_private_remote_processing: bool = False) -> str:
    """Always destroy the registered rental after bounded local work or failure."""
    _lease_ready(instance_id)
    try:
        _deadline_ready()
        _timer_ready()
        if not 1 <= pilot_rows <= 100 or not 1 <= workers <= 16:
            raise ValueError("Pilot rows must be 1-100 and LM workers 1-16")
        env = os.environ.copy()
        env["OLLAMA_HOST"] = "http://127.0.0.1:11435"
        env["EMBED_MODEL"] = "qwen3-embedding:4b"
        env["LM_STUDIOS_HOST"] = base
        settings = load_settings(ROOT / CONFIG)
        result = "pilot_only"
        _preflight(base)
        if not allow_private_remote_processing:
            raise PermissionError("Remote processing of private transcript prompts requires explicit approval")
        _pilot(settings, env, pilot_rows, workers)
        if not pilot_only:
            result = str(_full(settings, env, workers))
        return result
    finally:
        closed = vast_lease.check(LEASE, close=True, expected_id=instance_id)
        if closed["status"] != "destroyed_and_verified":
            raise RuntimeError("Vast instance was not confirmed destroyed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--lm-base", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--pilot-rows", type=int, default=20)
    parser.add_argument("--lm-workers", type=int, default=4)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--allow-private-remote-processing", action="store_true",
                        help="Only after the researcher explicitly approves processing on Vast.ai")
    args = parser.parse_args(argv)
    print(json.dumps({"local_result": run(args.instance_id, args.lm_base,
                                          pilot_rows=args.pilot_rows,
                                          workers=args.lm_workers,
                                          pilot_only=args.pilot_only,
                                          allow_private_remote_processing=args.allow_private_remote_processing)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
