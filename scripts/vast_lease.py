"""Optional independent Vast.ai lease watchdog with explicit host configuration.

Good: register one approved instance and run check from a configured timer.
Bad: infer approval to rent, process private prompts remotely, or destroy an
instance that is not named in this lease. No provider credentials are bundled.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEASE = Path(os.environ.get("VAST_LEASE_PATH", ROOT / "artifacts/vast-granite-lease.json"))
CLI = os.environ.get("VAST_LEASE_CLI", "vastai")
HOST = os.environ.get("VAST_LEASE_SSH_HOST", "")


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _cli(*args: str) -> str:
    """Run configured provider CLI; never send a secret argument."""
    if os.environ.get("VAST_LEASE_CLI_MODE") == "local":
        command = [CLI, *args]
    else:
        if not HOST:
            raise RuntimeError("Set VAST_LEASE_SSH_HOST or explicitly select VAST_LEASE_CLI_MODE=local")
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, CLI, *args]
    completed = subprocess.run(
        command,
        capture_output=True, text=True, timeout=45, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Vast CLI {args[0]} failed (exit {completed.returncode}): "
                           f"{completed.stderr[:250]}")
    return completed.stdout


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".part")
    with staged.open("w", encoding="utf-8") as out:
        os.fchmod(out.fileno(), 0o600)
        json.dump(value, out, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    staged.replace(path)


def register(path: Path, instance_id: int, hours: float, *, now: float | None = None) -> dict:
    """Bind one paid instance to an unambiguous hard destruction deadline."""
    if not 0 < instance_id or not 0 < hours <= 24:
        raise ValueError("Positive instance ID and a lease of at most 24 hours required")
    with _locked(path):
        if path.exists():
            raise FileExistsError("Existing lease must be closed and verified before another rental")
        start = time.time() if now is None else now
        lease = {"instance_id": instance_id, "created_epoch": start,
                 "destroy_at_epoch": start + hours * 3600,
                 "provider": "vast.ai", "host": HOST or "local"}
        _write(path, lease)
        return lease


def _active(instance_id: int) -> bool:
    next_token: str | None = None
    seen_tokens: set[str] = set()
    seen_ids: set[int] = set()
    total: int | None = None
    for _ in range(100):
        args = ("show", "instances-v1", "--raw")
        if next_token:
            args += ("--next-token", next_token)
        payload = json.loads(_cli(*args))
        if isinstance(payload, list):
            if next_token:
                raise ValueError("Vast pagination changed response shape")
            return any(int(row.get("id", -1)) == instance_id for row in payload)
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise ValueError("Unsuccessful Vast instances response; lease retained")
        rows = payload.get("instances")
        if not isinstance(rows, list):
            raise ValueError("Missing Vast instances list; lease retained")
        reported = payload.get("total_instances")
        if not isinstance(reported, int) or reported < 0:
            raise ValueError("Missing total_instances; lease retained")
        if total is None:
            total = reported
        elif total != reported:
            raise RuntimeError("Vast list changed during pagination; lease retained")
        for row in rows:
            instance = int(row["id"])
            if instance in seen_ids:
                raise RuntimeError("Duplicate Vast instance across pages; lease retained")
            seen_ids.add(instance)
        if instance_id in seen_ids:
            return True
        next_token = payload.get("next_token")
        if not next_token:
            if len(seen_ids) != total:
                raise RuntimeError("Vast list incomplete without next_token; lease retained")
            return False
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise RuntimeError("Invalid/repeated Vast pagination token; lease retained")
        seen_tokens.add(next_token)
    raise RuntimeError("Vast pagination exceeds 100 pages; lease retained")


def check(path: Path, *, now: float | None = None, close: bool = False,
          expected_id: int | None = None) -> dict:
    """Destroy expired/completed single instance; retain lease on any error."""
    with _locked(path):
        return _check_unlocked(path, now=now, close=close, expected_id=expected_id)


def _check_unlocked(path: Path, *, now: float | None, close: bool,
                    expected_id: int | None) -> dict:
    if not path.is_file():
        return {"status": "no_lease"}
    lease = json.loads(path.read_text(encoding="utf-8"))
    instance_id = int(lease["instance_id"])
    if instance_id <= 0 or lease.get("provider") != "vast.ai":
        raise ValueError("Invalid lease: refuse any provider mutation")
    if expected_id is not None and expected_id != instance_id:
        raise ValueError("Instance ID does not match registered lease")
    current = time.time() if now is None else now
    if not close and current < float(lease["destroy_at_epoch"]):
        return {"status": "leased", "instance_id": instance_id,
                "seconds_remaining": int(lease["destroy_at_epoch"] - current)}
    if _active(instance_id):
        _cli("destroy", "instance", str(instance_id), "-y")
    if _active(instance_id):
        raise RuntimeError(f"Instance {instance_id} still appears active; lease retained for retry")
    path.unlink()
    return {"status": "destroyed_and_verified", "instance_id": instance_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "check", "close"))
    parser.add_argument("--lease", type=Path, default=LEASE)
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--hours", type=float)
    args = parser.parse_args(argv)
    if args.action == "register":
        if args.instance_id is None or args.hours is None:
            parser.error("register requires --instance-id and --hours")
        result = register(args.lease, args.instance_id, args.hours)
    else:
        result = check(args.lease, close=args.action == "close", expected_id=args.instance_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
