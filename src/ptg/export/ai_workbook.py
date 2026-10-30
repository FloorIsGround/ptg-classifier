"""Copy a coded workbook and insert saved AI codes beside two human columns.

Good: export_profile('/private/coding.private.json') reads one frozen arm/repeat.
Bad: infer new decisions, choose an arm by score, or save over the source workbook.
No model adapter, service, codebook file, or network client is used.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from datetime import datetime, timezone
import zipfile

from ptg.export._ai_workbook_ooxml import (
    NS, ExportError, cell_text, column_name, raw_zip, read_cells, require,
    sheet_parts, shift_ref, transform, xml,
)

VERSION = "ai-workbook-export/1"
LIVE = "granite-live-openai/3"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _json(data):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "Duplicate JSON key")
            obj[key] = value
        return obj
    try:
        return json.loads(data, object_pairs_hook=unique,
                          parse_constant=lambda _: require(False, "Non-finite JSON number"))
    except (ValueError, UnicodeError):
        raise ExportError("Invalid JSON input") from None


def _path(value, base):
    require(isinstance(value, str) and bool(value.strip()), "Profile paths must be nonempty strings")
    path = base / value
    # Do not resolve away symlinks before examining them, including parent components.
    for part in [*reversed(path.parents), path]:
        require(not part.is_symlink(), "Symlink paths are refused")
    return Path(os.path.abspath(path))


def _stamp(path):
    info = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            "Inputs must be regular files with one hard link")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(path):
    before = _stamp(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) == before,
                "Input changed before being opened")
        data = stream.read()
    require(_stamp(path) == before, "Input changed while being read")
    return data


def _directory(path):
    require(not any((p / ".git").exists() for p in [path, *path.parents]),
            "Output directory must be outside Git checkouts")
    missing = []
    for part in [path, *path.parents]:
        if part.exists():
            break
        missing.append(part)
    for part in reversed(missing):
        part.mkdir(mode=0o700)
    info = path.stat(follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o700,
            "Output directory must be owned by you with mode 0700")


def _mapping(plan, workbook_sha):
    bundle = plan["bundle"]
    require(digest(bundle) == plan["bundle_sha256"], "Frozen bundle hash differs")
    require(digest(plan["population"]) == plan["population_sha256"], "Frozen population hash differs")
    conversion = bundle["conversion"]
    require(conversion["source_file_sha256"] == workbook_sha, "Workbook SHA256 differs from frozen conversion")
    mapping = conversion["mapping"]
    require(digest(mapping) == conversion["mapping_digest"], "Frozen mapping hash differs")
    require(mapping["schema"] == "xlsx-mapping/3" and mapping["row_limit"] is None,
            "A full frozen xlsx-mapping/3 panel is required")
    labels = bundle["labels"]
    require(isinstance(labels, list) and labels and all(isinstance(x, str) and x.strip() == x
            and x and not any(c in x for c in ",\n\r\x00") and not x.startswith("[AI INCOMPLETE")
            for x in labels) and len(set(labels)) == len(labels), "Invalid active label order")
    require(mapping["labels"] == labels and [e["label"] for e in bundle["codebook"]] == labels,
            "Frozen label/codebook order differs")
    specs = mapping["sheets"]
    require(isinstance(specs, list) and specs, "Frozen sheet mapping is missing")
    require(len({s["sheet"].casefold() for s in specs}) == len(specs), "Duplicate sheet mapping")
    for spec in specs:
        # This exporter intentionally accepts development targets only.
        name = "".join(spec["sheet"].casefold().split())
        require(spec["role"] == "target" and spec["semester"] in {"S26", "SP26"}
                and "fa26" not in name and "fall2026" not in name,
                "Only explicitly mapped S26/SP26 development targets are supported")
        columns = [spec["transcript"], spec["gold"], *spec["coders"]]
        require(len(spec["coders"]) == 2 and all(type(c) is int and 1 <= c < 16384 for c in columns)
                and len(set(columns)) == 4 and spec["coders"][1] == spec["coders"][0] + 1
                and type(spec["header_row"]) is int and 1 <= spec["header_row"] < 1048576,
                "Mapping requires physical transcript/gold columns and two adjacent human columns")
        require(spec["transcript"] < spec["coders"][0], "Transcript must precede human columns")
    return labels, specs


def _rows(plan, specs, archive):
    bundle = plan["bundle"]
    strings = []
    if "xl/sharedStrings.xml" in archive.namelist():
        strings = ["".join(t.text or "" for t in si.iter("{" + NS + "}t"))
                   for si in xml(archive.read("xl/sharedStrings.xml"))]
    parts = dict(sheet_parts(archive))
    by_name = {s["sheet"]: s for s in specs}
    cells, physical, expected = {}, {}, set()
    for spec in specs:
        require(spec["sheet"] in parts, "Mapped sheet is absent")
        name = spec["sheet"]
        cells[name], physical[name] = read_cells(archive.read(parts[name]))
        require(spec["header_row"] in physical[name], "Header row is absent")
        for pos in physical[name]:
            if pos > spec["header_row"]:
                text = cell_text(cells[name].get(column_name(spec["transcript"]) + str(pos)), strings)
                if text.strip():
                    expected.add((name, pos))
        for col in spec["coders"]:
            require(cells[name].get(column_name(col) + str(spec["header_row"])) is not None,
                    "Mapped human header cell is absent")
    seen_ids, seen_positions, targets = set(), set(), {}
    for row in bundle["rows"]:
        name, pos = row["sheet"], row["position"]
        require(name in by_name and type(pos) is int and pos > by_name[name]["header_row"],
                "Frozen row is outside the selected mapping")
        require(row["id"] not in seen_ids and (name, pos) not in seen_positions,
                "Duplicate frozen row identity or physical position")
        spec = by_name[name]
        require(row["semester"] == spec["semester"] and row["meeting"] == spec["meeting"]
                and row["id"] == digest([name, spec["semester"], spec["meeting"], pos]),
                "Frozen row identity differs from physical mapping")
        require(type(row["target"]) is bool and isinstance(row["text"], str), "Invalid frozen row state")
        source = cell_text(cells[name].get(column_name(spec["transcript"]) + str(pos)), strings)
        require(source == row["text"], "Transcript differs at a mapped physical row")
        require(row["target"] == ((name, pos) in expected), "Frozen target selection differs from full nonempty panel")
        seen_ids.add(row["id"])
        seen_positions.add((name, pos))
        if row["target"]:
            targets[row["id"]] = row
    require(expected == {(r["sheet"], r["position"]) for r in targets.values()} and targets,
            "Frozen rows do not cover the full selected transcript panel")
    population = plan["population"]
    require(len(population) == len(targets) and {p["row"] for p in population} == set(targets),
            "Plan population differs from mapped targets")
    return targets


def _saved_results(journal, source, archive, workbook_sha, allow_partial):
    before = _stamp(journal)
    require(not any(Path(str(journal) + suffix).exists() for suffix in ("-wal", "-journal")),
            "Journal must be a frozen SQLite file without WAL or rollback journal")
    with sqlite3.connect(journal.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        plans = db.execute("SELECT id,body,sha256 FROM plan").fetchall()
        require(len(plans) == 1 and plans[0][0] == 1, "Exactly one frozen plan is required")
        _, body, plan_sha = plans[0]
        require(isinstance(body, bytes) and hashlib.sha256(body).hexdigest() == plan_sha,
                "Frozen plan hash differs")
        if "plan_sha256" in source:
            require(source["plan_sha256"] == plan_sha, "Profile plan pin differs")
        plan = _json(body)
        require(plan["schema"] == LIVE and plan["repeat"] == source["repeat"],
                "Frozen adapter version or explicit repeat differs")
        labels, specs = _mapping(plan, workbook_sha)
        targets = _rows(plan, specs, archive)
        arm, repeat = source["arm"], source["repeat"]
        require(arm in plan["recipes"], "Explicit arm is absent from frozen plan")
        recipe, runtime = plan["recipes"][arm], plan["runtime"]
        require(recipe["model"] == runtime["model"] and recipe["revision"] == runtime["revision"],
                "Arm and frozen runtime model identity differ")
        all_jobs, selected, pairs, request_hashes = set(), {}, set(), {}
        for job in plan["jobs"]:
            require(isinstance(job["id"], str) and job["id"] not in all_jobs, "Duplicate frozen job identity")
            all_jobs.add(job["id"])
            if job["arm"] != arm:
                continue
            require(job["row"] in targets and len(job["labels"]) == 1 and job["labels"][0] in labels,
                    "Selected job is outside frozen target/label panel")
            label = job["labels"][0]
            row = targets[job["row"]]
            pair = (job["row"], label)
            require(pair not in pairs, "Duplicate selected row/label job")
            require(job["id"] == digest([LIVE, repeat, arm, job["row"], label])
                    and job["request_sha256"] == digest(job["request"])
                    and job["request"]["model"] == runtime["model"]
                    and job["meeting"] == row["meeting"]
                    and job["state"]["target"] == row["text"].strip()
                    and [q["label"] for q in job["questions"]] == [label],
                    "Selected job is not bound to the frozen request/row/label")
            selected[job["id"]] = pair
            request_hashes[job["id"]] = job["request_sha256"]
            pairs.add(pair)
        require(pairs == {(r, label) for r in targets for label in labels},
                "Selected arm lacks a full planned row/label panel")
        decisions, snapshot, requests, begun, seen = {}, [], [], set(), set()
        for seq, job_id, kind, blob, sha in db.execute(
                "SELECT seq,job,kind,body,sha256 FROM event WHERE kind IN ('begin','result') ORDER BY seq"):
            require(job_id in all_jobs, "Request/result event refers to a job outside the frozen plan")
            if job_id not in selected:
                continue
            require(isinstance(blob, bytes) and hashlib.sha256(blob).hexdigest() == sha,
                    "Selected request/result event hash differs")
            result = _json(blob)
            if kind == "begin":
                require(job_id not in begun, "Duplicate selected request event")
                require(isinstance(result, dict) and result.get("request_sha256") == request_hashes[job_id]
                        and result.get("runtime_id") == runtime["runtime_id"],
                        "Selected request event is not bound to the frozen plan/runtime")
                begun.add(job_id)
                requests.append([seq, job_id, sha])
                continue
            require(job_id not in seen, "Duplicate selected result event")
            require(job_id in begun, "Selected result lacks its preceding frozen request event")
            seen.add(job_id)
            require(isinstance(result, dict) and set(result) == {"outcome", "prediction"},
                    "Invalid selected result schema")
            outcome, prediction = result["outcome"], result["prediction"]
            require((outcome == "parsed" and type(prediction) is int and prediction in (0, 1))
                    or (outcome in {"abstained", "runtime_drift", "http_error", "invalid_response"}
                        and prediction is None), "Invalid selected result prediction")
            decisions[selected[job_id]] = (outcome, prediction)
            snapshot.append([seq, job_id, sha])
    require(_stamp(journal) == before, "Journal changed during export")
    values = {s["sheet"]: {} for s in specs}
    ai_columns = {s["sheet"]: s["coders"][1] + 1 for s in specs}
    rows, complete, parsed = [], 0, 0
    for row_id, row in targets.items():
        positive, missing = [], []
        for label in labels:
            outcome, prediction = decisions.get((row_id, label), ("missing", None))
            if outcome == "parsed":
                parsed += 1
                if prediction == 1:
                    positive.append(label)
            else:
                missing.append({"label": label, "status": outcome})
        if not missing:
            complete += 1
        text = (f"[AI INCOMPLETE: {len(labels) - len(missing)}/{len(labels)} labels; see sidecar]"
                if missing else ", ".join(positive))
        values[row["sheet"]][row["position"]] = text
        rows.append({"row_id": row_id, "sheet": row["sheet"], "physical_row": row["position"],
                     "ai_cell": column_name(ai_columns[row["sheet"]]) + str(row["position"]),
                     "status": "incomplete" if missing else "complete", "positive_labels": positive,
                     "unavailable": missing})
    require(allow_partial or complete == len(targets),
            "Selected panel is incomplete; use --allow-partial only for a visibly marked export")
    identity = {"kind": "granite-live", "adapter": LIVE, "arm": arm, "repeat": repeat,
                "plan_sha256": plan_sha, "bundle_sha256": plan["bundle_sha256"],
                "population_sha256": plan["population_sha256"],
                "model": runtime["model"], "model_revision": runtime["revision"],
                "runtime_id": runtime["runtime_id"], "result_snapshot_sha256": digest(snapshot),
                "request_snapshot_sha256": digest(requests), "selected_request_events": len(requests),
                "selected_result_events": len(snapshot), "last_selected_result_seq": max(
                    (x[0] for x in snapshot), default=None)}
    counts = {"sheets": len(specs), "rows": len(targets), "labels": len(labels),
              "expected_decisions": len(targets) * len(labels), "parsed_decisions": parsed,
              "complete_rows": complete, "incomplete_rows": len(targets) - complete,
              "no_code_rows": sum(not r["positive_labels"] and r["status"] == "complete" for r in rows)}
    return specs, labels, values, rows, identity, counts


def _verify(source, candidate, changes, specs, values):
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(candidate) as copied:
        require(original.namelist() == copied.namelist() and copied.testzip() is None,
                "Copied ZIP validation failed")
        for name in original.namelist():
            require(copied.read(name) == changes.get(name, original.read(name)),
                    "Copied package member differs from staged transformation")
        parts = dict(sheet_parts(copied))
        for spec in specs:
            before, _ = read_cells(original.read(parts[spec["sheet"]]))
            after, _ = read_cells(copied.read(parts[spec["sheet"]]))
            insert = spec["coders"][1] + 1
            for ref, cell in before.items():
                shifted = shift_ref(ref, insert)
                require(shifted in after, "Copied physical cell was lost")
                expected = dict(cell.attrib, r=shifted)
                require(after[shifted].attrib == expected, "Original cell attributes changed")
                # Formula text may change; all other original cell content must survive.
                for old, new in zip(cell, after[shifted], strict=True):
                    if old.tag != "{" + NS + "}f":
                        require(ET_bytes(old) == ET_bytes(new), "Original cell content changed")
            for pos, text in {**values[spec["sheet"]], spec["header_row"]: "AI"}.items():
                ref = column_name(insert) + str(pos)
                require(ref in after and cell_text(after[ref], []) == text,
                        "AI cell verification failed")
        return len(original.namelist()) - len(changes)


def ET_bytes(element):
    from xml.etree import ElementTree
    return ElementTree.tostring(element)


def _publish(staged, destination):
    # Linux renameat2 is atomic and refuses every existing destination, including links.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    require(rename is not None, "Atomic non-overwriting export requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise ExportError("Destination already exists; choose a new output filename")
        raise ExportError("Atomic non-overwriting publication failed")


def export_profile(profile_path, *, output=None, allow_partial=False):
    """Export a new private workbook plus an identity/coverage/row-mapping sidecar.

    Good: export_profile('/private/review.private.json') with a new output path.
    Bad: pass the original workbook as output, or treat an incomplete panel as no code.
    Relative profile paths always resolve from the profile's directory.
    """
    try:
        return _export_profile(profile_path, output=output, allow_partial=allow_partial)
    except ExportError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            sqlite3.Error, zipfile.BadZipFile, OverflowError):
        raise ExportError("Export failed: unreadable input or unsupported input structure; no cell values logged") from None


def _export_profile(profile_path, *, output, allow_partial):
    profile_path = _path(str(profile_path), Path.cwd())
    profile_data = _read(profile_path)
    profile = _json(profile_data)
    require(isinstance(profile, dict) and set(profile) == {"schema", "workbook", "output", "source"}
            and profile["schema"] == "ai-workbook-profile/1", "Invalid ai-workbook-profile/1 profile")
    source = profile["source"]
    require(isinstance(source, dict) and set(source) in (
        {"kind", "journal", "arm", "repeat"}, {"kind", "journal", "arm", "repeat", "plan_sha256"})
        and source["kind"] == "granite-live" and all(isinstance(source[k], str) and source[k].strip()
        for k in ("journal", "arm", "repeat")), "Source requires granite-live journal, exact arm, and repeat")
    workbook = _path(profile["workbook"], profile_path.parent)
    journal = _path(source["journal"], profile_path.parent)
    destination = _path(str(output) if output is not None else profile["output"], profile_path.parent)
    sidecar = Path(str(destination) + ".ai.json")
    require(destination.suffix.lower() == ".xlsx", "Output filename must end in .xlsx")
    require(not os.path.lexists(destination) and not os.path.lexists(sidecar),
            "Destination or sidecar already exists; choose a new output filename")
    _directory(destination.parent)
    source_bytes = _read(workbook)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    journal_stamp = _stamp(journal)
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate ZIP members")
        specs, labels, values, rows, identity, counts = _saved_results(
            journal, source, archive, source_sha, allow_partial)
        changes = transform(archive, specs, values)
    report = {"schema": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
              "profile_sha256": hashlib.sha256(profile_data).hexdigest(),
              "workbook_sha256": source_sha, "source": identity, "active_labels": labels,
              "status": "partial" if counts["incomplete_rows"] else "complete",
              "allow_partial": allow_partial, "counts": counts,
              "mapping": [{**s, "ai_column": s["coders"][1] + 1} for s in specs],
              "rows": rows, "changed_zip_members": sorted(changes), "inference_calls": 0}
    with tempfile.TemporaryDirectory(prefix=".ai-export-", dir=destination.parent) as staging:
        stage = Path(staging)
        staged_workbook, staged_sidecar = stage / "copy.xlsx", stage / "copy.ai.json"
        fd = os.open(staged_workbook, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw_zip(source_bytes, changes))
            stream.flush()
            os.fsync(stream.fileno())
        report["unchanged_zip_members"] = _verify(source_bytes, staged_workbook, changes, specs, values)
        report["output_sha256"] = hashlib.sha256(staged_workbook.read_bytes()).hexdigest()
        require(hashlib.sha256(_read(workbook)).hexdigest() == source_sha
                and _stamp(journal) == journal_stamp, "Source workbook or journal changed during export")
        fd = os.open(staged_sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish(staged_sidecar, sidecar)
        try:
            _publish(staged_workbook, destination)
        except BaseException:
            sidecar.unlink()
            raise
    return {"status": report["status"], "counts": counts, "inference_calls": 0}


def main(argv=None):
    """Good: --profile /private/review.private.json. Bad: a long per-run flag list."""
    parser = argparse.ArgumentParser(description="Copy a workbook and insert AI codes from frozen saved results.")
    parser.add_argument("--profile", required=True, type=Path, help="Reusable local .private.json profile")
    parser.add_argument("--output", type=Path, help="New filename, relative to the profile directory")
    parser.add_argument("--allow-partial", action="store_true", help="Mark incomplete rows visibly and in the sidecar")
    args = parser.parse_args(argv)
    try:
        result = export_profile(args.profile, output=args.output, allow_partial=args.allow_partial)
    except ExportError as exc:
        print("Export refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0
