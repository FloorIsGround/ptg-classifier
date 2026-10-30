"""Synthetic saved-result-to-workbook workflows. No private fixture material."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile

import openpyxl
from openpyxl.comments import Comment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.workbook.defined_name import DefinedName
import pytest

from ptg.export.ai_workbook import LIVE, ExportError, canonical, digest, export_profile
from ptg.export._ai_workbook_ooxml import raw_zip

ROOT = Path(__file__).resolve().parents[1]
SHEET = "Fictional Team (S26)"
LABELS = ["Planning", "Reasoning", "Communication"]
SENTINEL = "Fictional private utterance: do not log me"


def replace_parts(path, changes):
    path.write_bytes(raw_zip(path.read_bytes(), changes))


def fixture_files(tmp_path):
    """Good: build fictional data locally. Bad: copy a real saved plan into tests."""
    tmp_path.chmod(0o700)
    workbook = tmp_path / "input.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET
    ws.append(["A", "B", "Speaker", "Transcript", "Coder", "Coder", "Reconciled",
               "Existing right", "I", "J", "K", "Transcript"])
    ws.append([None, None, "Fictional", SENTINEL, "Planning", "", "Planning", "right value",
               None, None, None, "decoy text"])
    ws.append([None, None, "Fictional", None, None, None, None, "blank transcript row"])
    ws.append([None, None, "Fictional", 42, None, "", "", "right numeric"])
    ws["G2"].comment = Comment("Fictional comment", "Fictional author")
    ws["E2"].font = openpyxl.styles.Font(bold=True)
    ws["F2"].number_format = "@"
    ws.column_dimensions["F"].width = 35
    ws.column_dimensions["G"].width = 42
    ws.auto_filter.ref = "C1:H4"
    ws.auto_filter.add_filter_column(4, ["Planning"])
    validation = DataValidation(type="list", formula1='"Planning,Reasoning,Communication"')
    validation.add("E2:G4")
    ws.add_data_validation(validation)
    ws.freeze_panes = "G2"
    ws["I2"] = "=G2&\"G2\""
    ws["J2"] = "=SUM(G4:H4)"
    other = wb.create_sheet("Other")
    other["A1"] = "='Fictional Team (S26)'!$G$2"
    other["A2"] = "=A1"
    other["B1"] = "unchanged elsewhere"
    untouched = wb.create_sheet("Unselected")
    untouched["A1"] = "unselected bytes"
    wb.defined_names.add(DefinedName("ReviewCodes", attr_text="'Fictional Team (S26)'!$G$2:$G$4"))
    wb.save(workbook)
    with zipfile.ZipFile(workbook, "a") as z:
        z.writestr("custom/opaque.bin", b"untouched opaque payload\x00\xff")
    spec = {"sheet": SHEET, "semester": "S26", "meeting": "fictional-meeting",
            "role": "target", "header_row": 1, "transcript": 4, "coders": [5, 6], "gold": 7}
    mapping = {"schema": "xlsx-mapping/3", "row_limit": None, "labels": LABELS, "sheets": [spec]}
    rows = []
    for position, text, target in [(2, SENTINEL, True), (3, "", False), (4, "42", True)]:
        rows.append({"id": digest([SHEET, "S26", "fictional-meeting", position]),
                     "sheet": SHEET, "semester": "S26", "meeting": "fictional-meeting",
                     "position": position, "text": text, "target": target})
    bundle = {"labels": LABELS, "rows": rows, "codebook": [{"label": l} for l in LABELS],
              "conversion": {"source_file_sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                             "mapping": mapping, "mapping_digest": digest(mapping)}}
    runtime = {"model": "fictional-model", "revision": "a" * 40, "runtime_id": "fictional-runtime"}
    jobs = []
    for arm in ["rag_baseline", "another_arm"]:
        for row in rows:
            if not row["target"]:
                continue
            for label in LABELS:
                request = {"model": "fictional-model", "messages": []}
                jobs.append({"id": digest([LIVE, "r1", arm, row["id"], label]), "arm": arm,
                             "row": row["id"], "labels": [label], "meeting": row["meeting"],
                             "request": request, "request_sha256": digest(request),
                             "state": {"target": row["text"]}, "questions": [{"label": label}]})
    population = [{"row": r["id"]} for r in rows if r["target"]]
    plan = {"schema": LIVE, "repeat": "r1", "bundle": bundle, "bundle_sha256": digest(bundle),
            "population": population, "population_sha256": digest(population), "runtime": runtime,
            "recipes": {arm: {"model": runtime["model"], "revision": runtime["revision"]}
                        for arm in ["rag_baseline", "another_arm"]}, "jobs": jobs}
    journal = tmp_path / "live.sqlite3"
    with sqlite3.connect(journal) as db:
        db.executescript("CREATE TABLE plan(id INTEGER,body BLOB,sha256 TEXT);"
                         "CREATE TABLE event(seq INTEGER PRIMARY KEY,job TEXT,kind TEXT,body BLOB,sha256 TEXT);")
        body = canonical(plan)
        db.execute("INSERT INTO plan VALUES(1,?,?)", (body, hashlib.sha256(body).hexdigest()))
        for seq, job in enumerate(jobs, 1):
            begin = canonical({"request_sha256": job["request_sha256"], "runtime_id": runtime["runtime_id"]})
            db.execute("INSERT INTO event VALUES(?,?,?,?,?)", (
                seq * 2 - 1, job["id"], "begin", begin, hashlib.sha256(begin).hexdigest()))
            # Two positives in frozen label order, then a genuinely all-negative numeric row.
            prediction = int(job["row"] == rows[0]["id"] and job["labels"][0] in LABELS[:2])
            if job["arm"] == "another_arm":
                prediction = 1
            result = canonical({"outcome": "parsed", "prediction": prediction})
            db.execute("INSERT INTO event VALUES(?,?,?,?,?)", (
                seq * 2, job["id"], "result", result, hashlib.sha256(result).hexdigest()))
    profile = tmp_path / "review.private.json"
    profile.write_text(json.dumps({"schema": "ai-workbook-profile/1", "workbook": workbook.name,
                                  "output": "exports/copied.xlsx", "source": {
                                      "kind": "granite-live", "journal": journal.name,
                                      "arm": "rag_baseline", "repeat": "r1"}}))
    profile.chmod(0o600)
    return profile, workbook, journal, plan


def update_plan(journal, plan):
    bundle = plan["bundle"]
    bundle["conversion"]["mapping_digest"] = digest(bundle["conversion"]["mapping"])
    plan["bundle_sha256"] = digest(bundle)
    plan["population_sha256"] = digest(plan["population"])
    body = canonical(plan)
    with sqlite3.connect(journal) as db:
        db.execute("UPDATE plan SET body=?,sha256=?", (body, hashlib.sha256(body).hexdigest()))


def sync_workbook_hash(workbook, journal, plan):
    plan["bundle"]["conversion"]["source_file_sha256"] = hashlib.sha256(workbook.read_bytes()).hexdigest()
    update_plan(journal, plan)


def read_report(profile):
    return json.loads((profile.parent / "exports/copied.xlsx.ai.json").read_text())


def test_cli_copies_physical_rows_and_preserves_workbook(tmp_path):
    profile, workbook, _, _ = fixture_files(tmp_path)
    before = workbook.read_bytes()
    # Invoke from another cwd: all profile-relative paths must still work.
    result = subprocess.run([sys.executable, str(ROOT / "scripts/export_ai_workbook.py"),
                             "--profile", str(profile)], cwd=tmp_path.parent,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert SENTINEL not in result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["counts"]["rows"] == 2
    assert summary["counts"]["no_code_rows"] == 1
    assert summary["inference_calls"] == 0
    copied = tmp_path / "exports/copied.xlsx"
    wb = openpyxl.load_workbook(copied)
    ws = wb[SHEET]
    assert [ws.cell(1, col).value for col in (5, 6, 7, 8, 13)] == [
        "Coder", "Coder", "AI", "Reconciled", "Transcript"]
    assert ws["G2"].value == "Planning, Reasoning"
    assert ws["G3"].value is None
    assert ws["G4"].value in ("", None)
    assert ws["D4"].value == 42
    assert ws["I2"].value == "right value"
    assert ws["J2"].value == '=H2&"G2"'
    assert ws["K2"].value == "=SUM(H4:I4)"
    assert ws["E2"].font.bold and ws["F2"].number_format == "@"
    assert ws["H2"].comment.text == "Fictional comment"
    assert str(ws.data_validations.dataValidation[0].sqref) == "E2:H4"
    assert ws.auto_filter.ref == "C1:I4"
    assert ws.auto_filter.filterColumn[0].colId == 5
    assert ws.freeze_panes == "H2"
    assert ws.column_dimensions["H"].width == 42
    assert ws.column_dimensions["G"].width == 35
    assert wb["Other"]["A1"].value == "='Fictional Team (S26)'!$H$2"
    assert wb.defined_names["ReviewCodes"].attr_text == "'Fictional Team (S26)'!$H$2:$H$4"
    assert workbook.read_bytes() == before
    report = read_report(profile)
    assert report["source"]["arm"] == "rag_baseline" and report["source"]["repeat"] == "r1"
    assert [r["physical_row"] for r in report["rows"]] == [2, 4]
    assert SENTINEL not in json.dumps(report)
    assert copied.stat().st_mode & 0o777 == 0o600
    assert Path(str(copied) + ".ai.json").stat().st_mode & 0o777 == 0o600
    assert copied.parent.stat().st_mode & 0o777 == 0o700
    assert copied.stat().st_nlink == 1
    with zipfile.ZipFile(io.BytesIO(before)) as a, zipfile.ZipFile(copied) as b:
        assert a.namelist() == b.namelist()
        for name in a.namelist():
            if name not in report["changed_zip_members"]:
                assert a.read(name) == b.read(name)
                ia, ib = a.getinfo(name), b.getinfo(name)
                # The compressed bytes and original local headers survive as well.
                end_a = ia.header_offset + 30 + len(ia.filename.encode()) + len(ia.extra) + ia.compress_size
                end_b = ib.header_offset + 30 + len(ib.filename.encode()) + len(ib.extra) + ib.compress_size
                assert before[ia.header_offset:end_a] == copied.read_bytes()[ib.header_offset:end_b]


@pytest.mark.parametrize("mutation,match", [
    ("workbook", "Workbook SHA256"), ("row", "Transcript differs"),
    ("position", "row identity"), ("duplicate_row", "Duplicate frozen row"),
    ("duplicate_job", "Duplicate frozen job"), ("missing_job", "full planned"),
    ("wrong_job", "not bound"), ("duplicate_result", "Duplicate selected result"),
    ("orphan_result", "outside the frozen plan"), ("bad_prediction", "Invalid selected result"),
    ("bool_prediction", "Invalid selected result"), ("result_hash", "event hash"),
    ("plan_hash", "plan hash"), ("repeat", "explicit repeat"),
    ("holdout", "development targets"), ("bundle_hash", "bundle hash"),
    ("missing_begin", "preceding frozen request"), ("wrong_begin", "not bound"),
    ("duplicate_begin", "Duplicate selected request"),
])
def test_rejects_untrusted_or_mismatched_inputs(tmp_path, mutation, match):
    profile, workbook, journal, plan = fixture_files(tmp_path)
    if mutation == "workbook":
        workbook.write_bytes(workbook.read_bytes() + b"changed")
    elif mutation in {"row", "position", "duplicate_row"}:
        if mutation == "row":
            plan["bundle"]["rows"][0]["text"] = "different fictional text"
        elif mutation == "position":
            plan["bundle"]["rows"][0]["position"] = 3
        else:
            plan["bundle"]["rows"].append(plan["bundle"]["rows"][0])
        update_plan(journal, plan)
    elif mutation in {"duplicate_job", "missing_job", "wrong_job"}:
        if mutation == "duplicate_job":
            plan["jobs"].append(plan["jobs"][0])
        elif mutation == "missing_job":
            del plan["jobs"][0]
        else:
            plan["jobs"][0]["id"] = "arbitrary-job"
        update_plan(journal, plan)
    elif mutation == "repeat":
        value = json.loads(profile.read_text()); value["source"]["repeat"] = "r2"
        profile.write_text(json.dumps(value))
    elif mutation == "holdout":
        plan["bundle"]["conversion"]["mapping"]["sheets"][0]["semester"] = "FA26"
        update_plan(journal, plan)
    else:
        with sqlite3.connect(journal) as db:
            if mutation == "duplicate_result":
                db.execute("INSERT INTO event(job,kind,body,sha256) SELECT job,kind,body,sha256 FROM event WHERE seq=2")
            elif mutation == "orphan_result":
                db.execute("UPDATE event SET job='orphan' WHERE seq=2")
            elif mutation in {"bad_prediction", "bool_prediction"}:
                body = canonical({"outcome": "parsed", "prediction": True if mutation == "bool_prediction" else 2})
                db.execute("UPDATE event SET body=?,sha256=? WHERE seq=2", (body, hashlib.sha256(body).hexdigest()))
            elif mutation == "result_hash":
                db.execute("UPDATE event SET sha256='bad' WHERE seq=2")
            elif mutation == "plan_hash":
                db.execute("UPDATE plan SET sha256='bad'")
            elif mutation == "bundle_hash":
                plan["bundle_sha256"] = "bad"
                body = canonical(plan)
                db.execute("UPDATE plan SET body=?,sha256=?", (body, hashlib.sha256(body).hexdigest()))
            elif mutation == "missing_begin":
                db.execute("DELETE FROM event WHERE seq=1")
            elif mutation == "wrong_begin":
                body = canonical({"request_sha256": "wrong", "runtime_id": "wrong"})
                db.execute("UPDATE event SET body=?,sha256=? WHERE seq=1", (body, hashlib.sha256(body).hexdigest()))
            elif mutation == "duplicate_begin":
                db.execute("INSERT INTO event(job,kind,body,sha256) SELECT job,kind,body,sha256 FROM event WHERE seq=1")
    with pytest.raises(ExportError, match=match) as exc:
        export_profile(profile)
    assert SENTINEL not in str(exc.value)
    assert not (tmp_path / "exports/copied.xlsx").exists()
    assert not (tmp_path / "exports/copied.xlsx.ai.json").exists()


@pytest.mark.parametrize("missing", [True, False])
def test_partial_is_explicit_visible_and_never_negative(tmp_path, missing):
    profile, _, journal, _ = fixture_files(tmp_path)
    with sqlite3.connect(journal) as db:
        if missing:
            db.execute("DELETE FROM event WHERE seq=8")
        else:
            body = canonical({"outcome": "abstained", "prediction": None})
            db.execute("UPDATE event SET body=?,sha256=? WHERE seq=8", (body, hashlib.sha256(body).hexdigest()))
    with pytest.raises(ExportError, match="incomplete"):
        export_profile(profile)
    result = export_profile(profile, allow_partial=True)
    assert result["status"] == "partial" and result["counts"]["no_code_rows"] == 0
    ws = openpyxl.load_workbook(tmp_path / "exports/copied.xlsx")[SHEET]
    assert ws["G4"].value.startswith("[AI INCOMPLETE: 2/3")
    report = read_report(profile)
    assert report["rows"][1]["unavailable"][0]["status"] == ("missing" if missing else "abstained")


@pytest.mark.parametrize("kind", ["existing", "sidecar", "symlink", "hardlink", "parent_symlink", "git", "permissions"])
def test_non_overwriting_private_destination(tmp_path, kind):
    profile, workbook, _, _ = fixture_files(tmp_path)
    parent = tmp_path / "exports"
    parent.mkdir(mode=0o700)
    output = parent / "copied.xlsx"
    if kind == "existing":
        output.write_bytes(b"existing")
    elif kind == "sidecar":
        Path(str(output) + ".ai.json").write_bytes(b"existing")
    elif kind == "symlink":
        output.symlink_to(workbook)
    elif kind == "hardlink":
        os.link(workbook, output)
    elif kind == "parent_symlink":
        parent.rmdir(); parent.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "git":
        (tmp_path / ".git").mkdir()
    else:
        parent.chmod(0o755)
    before = workbook.read_bytes()
    with pytest.raises(ExportError):
        export_profile(profile)
    assert workbook.read_bytes() == before
    if kind == "existing":
        assert output.read_bytes() == b"existing"


def test_validation_failure_does_not_publish(tmp_path, monkeypatch):
    profile, _, _, _ = fixture_files(tmp_path)
    import ptg.export.ai_workbook as module
    def fail(*args):
        raise ExportError("Synthetic staged validation failure")
    monkeypatch.setattr(module, "_verify", fail)
    with pytest.raises(ExportError, match="staged validation"):
        export_profile(profile)
    assert list((tmp_path / "exports").iterdir()) == []


def test_publish_race_refuses_to_overwrite_and_removes_own_sidecar(tmp_path, monkeypatch):
    profile, _, _, _ = fixture_files(tmp_path)
    import ptg.export.ai_workbook as module
    original = module._publish
    def race(staged, destination):
        if destination.suffix == ".xlsx":
            destination.write_bytes(b"other writer")
        original(staged, destination)
    monkeypatch.setattr(module, "_publish", race)
    with pytest.raises(ExportError, match="already exists"):
        export_profile(profile)
    assert (tmp_path / "exports/copied.xlsx").read_bytes() == b"other writer"
    assert not (tmp_path / "exports/copied.xlsx.ai.json").exists()
