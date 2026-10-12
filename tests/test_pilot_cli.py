"""Pilot CLI must not write to or read from the full result namespace."""

from pathlib import Path

from ptg import cli
from ptg.pipeline import run


REPO = Path(__file__).resolve().parents[1]


def test_pilot_cli_isolates_infer_and_export(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "synthetic-embed")
    monkeypatch.setenv("LM_STUDIOS_HOST", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("LM_MODEL", "synthetic-lm")
    seen = []

    def infer(settings, **kwargs):
        seen.append(("infer", settings, kwargs))

    def export(settings, **kwargs):
        seen.append(("export", settings, kwargs))

    monkeypatch.setattr(run, "infer_sheet", infer)
    monkeypatch.setattr(run, "export_sheet", export)
    base = ["--config", str(REPO / "config/study-21.yaml")]
    target = ["--team", "1", "--meet", "1", "--semester", "S26"]
    cli.main(base + ["infer"] + target + ["--pilot-row-limit", "3"])
    cli.main(base + ["export"] + target + ["--pilot", "3"])
    assert len(seen) == 2
    assert all(settings.artifacts_dir == Path("artifacts/study21_core_pilot3")
               for _, settings, _ in seen)
    assert all(settings.index_artifacts_dir == Path("artifacts/f25_core_index")
               for _, settings, _ in seen)
    assert seen[0][2]["pilot_row_limit"] == 3
