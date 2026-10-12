"""Create a source-only Colab bundle; never include transcript data or .env files.

Good: uv run python scripts/create_colab_bundle.py --output artifacts/ptg-colab.zip
Bad: zip -r project.zip .  # would capture data, .env, caches, and results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXACT = (".python-version", "README.md", "pyproject.toml", "uv.lock", "config/default.yaml", "config/latest-local.yaml", "scripts/audit_workbook.py")
GLOBS = ("config/codebooks/*.yaml", "src/**/*.py")


def make_bundle(output: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    files = {root / name for name in EXACT}
    for pattern in GLOBS:
        files.update(root.glob(pattern))
    if not files or any(not file.is_file() or file.is_symlink() for file in files):
        raise ValueError("Missing or symlinked required source file")
    entries = sorted((file.relative_to(root).as_posix(), file.read_bytes()) for file in files)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *(name for name, _ in entries)],
        cwd=root, text=True,
    )
    manifest = {
        "base_commit": commit,
        "bundled_files_uncommitted": bool(status.strip()),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in entries},
    }
    output = output.resolve()
    if output in files:
        raise ValueError("Output would overwrite a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries + [("manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": len(entries), "base_commit": commit, "bundled_files_uncommitted": manifest["bundled_files_uncommitted"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ptg-colab.zip")
    print(json.dumps(make_bundle(parser.parse_args().output), indent=2))
