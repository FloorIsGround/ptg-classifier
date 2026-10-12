import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path


def test_bundle_contains_only_allowlisted_source(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bundle", root / "scripts/create_colab_bundle.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = tmp_path / "bundle.zip"
    report = module.make_bundle(dest, root)
    with zipfile.ZipFile(dest) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert set(zf.namelist()) == set(manifest["files"]) | {"manifest.json"}
        assert manifest["base_commit"] == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        assert isinstance(manifest["bundled_files_uncommitted"], bool)
        assert all(hashlib.sha256(zf.read(name)).hexdigest() == digest
                   for name, digest in manifest["files"].items())
        assert all(name in module.EXACT or name.startswith(("src/", "config/codebooks/"))
                   for name in manifest["files"])
        assert not any(".env" in name or name.startswith("data/") for name in zf.namelist())
        assert report["sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert {"config/latest-local.yaml", "scripts/audit_workbook.py"} <= set(manifest["files"])


def test_colab_notebook_is_valid_and_has_no_executed_cells():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/colab_runner.ipynb").read_text())
    assert notebook["nbformat"] == 4
    notebook_source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "config/latest-local.yaml" in notebook_source
    assert "scripts/audit_workbook.py" in notebook_source
    assert "FA26" in notebook_source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "colab_runner.ipynb", "exec")
