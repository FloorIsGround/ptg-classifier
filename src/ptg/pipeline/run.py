from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ptg.classify.binary_classifier import BinaryClassificationJob, classify_binary_jobs
from ptg.classify import binary_classifier
from ptg.classify.codebook import load_codebook
from ptg.config import Settings
from ptg.domain.models import CodebookEntry, SheetRef
from ptg.export.aggregate import aggregate_runs
from ptg.index.embed import embed_many
from ptg.ingest.excel import (
    load_sheet_df,
    normalize_sheet_df,
    sheet_name,
)
from ptg.storage.faiss_index import FaissIPIndex
from ptg.storage.files import artifact_paths
from ptg.storage.checkpoints import CheckpointStore
from ptg.storage.raw_results import RawResult, next_run_sheet_name, read_raw_results, write_raw_results
from ptg.util.codes import canonicalize_codes, parse_codes

log = logging.getLogger(__name__)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_manifest_path(settings: Settings) -> Path:
    return settings.index_artifacts_dir / f"{settings.run_prefix}index_manifest.json"


def _run_fingerprint(settings: Settings, source_sheet: str, lm_timeout: float,
                     row_indices: list[int]) -> str:
    """Reject resuming a checkpoint with different data, prompts or models."""
    payload = {
        "workbook": _file_sha256(settings.workbook_path),
        "index": json.loads(_index_manifest_path(settings).read_text(encoding="utf-8")),
        "codebook": _file_sha256(settings.codebook_path),
        "prompts": _file_sha256(Path(binary_classifier.__file__)),
        "embed_model": settings.embed_model,
        "lm_model": settings.lm_model,
        "lm_reasoning_effort": getattr(settings, "lm_reasoning_effort", None),
        "lm_api_base": settings.lm_api_base,
        "source_sheet": source_sheet,
        "row_indices": row_indices,
        "labels": list(settings.active_labels),
        "top_k": settings.retrieval_top_k,
        "prev_k": settings.prev_k,
        "max_retries": settings.binary_max_retries,
        "lm_timeout": lm_timeout,
    }
    # Preserve the previous default fingerprint for completed Qwen checkpoints.
    # Every non-default decoding control changes the fingerprint and blocks resume.
    for key, value, default in (
        ("lm_temperature", getattr(settings, "lm_temperature", 0.0), 0.0),
        ("lm_top_p", getattr(settings, "lm_top_p", None), None),
        ("lm_max_tokens", getattr(settings, "lm_max_tokens", 10), 10),
        ("lm_chat_template_kwargs", getattr(settings, "lm_chat_template_kwargs", None), None),
        ("lm_seed", getattr(settings, "lm_seed", None), None),
    ):
        if value != default:
            payload[key] = value
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _query_cache_fingerprint(transcripts: list[str], model: str) -> str:
    """Bind cached vectors to ordered source text and the exact embedding model."""
    payload = json.dumps({"model": model, "transcripts": transcripts}, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_target_not_indexed(metas: pd.DataFrame, target_sheet: str) -> None:
    """Prevent evaluation on a sheet whose own human labels are in retrieval."""
    if target_sheet in set(metas["sheet"].astype(str)):
        raise ValueError(f"Target sheet {target_sheet!r} is part of the index; refusing label leakage")


def _retrieved_rows(distances: np.ndarray, indices: np.ndarray, metas: list[dict]) -> list[tuple[dict, float]]:
    """Ignore FAISS's -1 padding instead of treating it as the last metadata row."""
    return [
        (dict(metas[int(idx)]), float(dist))
        for dist, idx in zip(distances, indices)
        if 0 <= int(idx) < len(metas)
    ]


def _codes_key(codes: list[str]) -> str:
    return "|".join(dict.fromkeys(codes))


def _prev_context_block(df: pd.DataFrame, i: int, k: int = 2) -> str:
    bits: list[str] = []
    for j in range(max(0, i - k), i):
        spk = str(df.at[j, "Speaker"]) if "Speaker" in df.columns else ""
        txt = str(df.at[j, "Transcript"]) if "Transcript" in df.columns else ""
        if txt.strip():
            bits.append(f"{spk.strip()}: {txt.strip()}")
    return "\n".join(bits)


def _chunks(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _timing_series(df_raw: pd.DataFrame) -> pd.Series:
    """Extract the timestamp column from the raw (pre-normalisation) sheet DataFrame.

    Tries named columns first, falls back to positional column B (index 1).
    """
    for col in ("Timestamp", "timestamp", "Time", "time"):
        if col in df_raw.columns:
            return df_raw[col].astype(str).fillna("").str.strip()
    if df_raw.shape[1] > 1:
        return df_raw.iloc[:, 1].astype(str).fillna("").str.strip()
    return pd.Series([""] * len(df_raw), index=df_raw.index)


def _format_binary_example(meta: dict, score: float) -> str:
    """Format a retrieved index row as a concise example for a binary prompt."""
    speaker = meta.get("speaker", "")
    transcript = meta.get("transcript", "")
    prev_context = meta.get("prev_context", "")
    lines = [f"Speaker: {speaker} [cos={score:.4f}]"]
    if prev_context:
        lines.append(f"Context: {prev_context}")
    lines.append(f"Transcript: {transcript}")
    return "\n".join(lines)


def build_index(
    settings: Settings,
    *,
    semester: str,
    teams: list[int],
    meets: list[int],
) -> None:
    """Build a FAISS cosine-similarity index and metas parquet from coded rows.

    Reads the Excel workbook at settings.workbook_path. Only indexes rows that
    have at least one reconciled code. Writes artifacts using settings.run_prefix.
    """
    settings.index_artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(settings.index_artifacts_dir, settings.run_prefix)
    codebook = load_codebook(settings.codebook_path)
    if not settings.embed_model:
        raise ValueError("An embedding model is required to build the index")
    workbook_digest = _file_sha256(settings.workbook_path)
    codebook_digest = _file_sha256(settings.codebook_path)
    excluded = {" ".join(x.casefold().split()) for x in getattr(settings, "excluded_source_labels", ())}
    excluded_instances = 0
    excluded_only_rows = 0

    embeddings: list[np.ndarray] = []
    metas: list[dict] = []
    source_sheets: list[str] = []

    for team in teams:
        for meet in meets:
            sheet = SheetRef(team=team, meet=meet, semester=semester)
            name = sheet_name(sheet, settings.sheet_template)

            # An explicit source selection must not quietly become a different
            # corpus when a sheet is missing or unreadable.
            _, df_raw = load_sheet_df(str(settings.workbook_path), sheet, settings.sheet_template)
            if name in source_sheets:
                raise ValueError(f"Duplicate source sheet selected: {name}")
            source_sheets.append(name)

            df = normalize_sheet_df(df_raw)

            if "Reconciled Codes" in df.columns:
                df["Reconciled Codes"] = df["Reconciled Codes"].astype("string").fillna("")
            else:
                df["Reconciled Codes"] = ""

            docs: list[str] = []
            row_payloads: list[dict] = []

            for i in df.index:
                transcript = str(df.at[i, "Transcript"]).strip()
                if not transcript:
                    continue

                speaker = str(df.at[i, "Speaker"]).strip()
                codes_raw = df.at[i, "Reconciled Codes"]
                raw_codes = parse_codes(codes_raw)
                kept_codes = [code for code in raw_codes
                              if " ".join(code.casefold().split()) not in excluded]
                excluded_instances += len(raw_codes) - len(kept_codes)
                if raw_codes and not kept_codes:
                    excluded_only_rows += 1
                # Every other unknown source code is a hard error, not a silent drop.
                codes = canonicalize_codes(kept_codes, codebook=codebook)

                if not codes:
                    continue

                prev_block = _prev_context_block(df, int(i), k=settings.prev_k)

                # Use the same retrieval surface as inference queries. Human codes
                # belong in metadata/prompt examples, not the vector text.
                doc = transcript
                docs.append(doc)
                row_payloads.append(
                    {
                        "sheet": name,
                        "row_index": int(i),
                        "team": team,
                        "meet": meet,
                        "speaker": speaker,
                        "transcript": transcript,
                        "prev_context": prev_block,
                        "codes": codes,
                        "code_ids": [codebook.entry_by_label(c).id for c in codes],
                        "codes_key": _codes_key(codes),
                    }
                )

            if not docs:
                continue

            log.info("Embedding %d coded rows from sheet %s", len(docs), name)
            for batch in _chunks(list(range(len(docs))), 32):
                batch_docs = [docs[i] for i in batch]
                batch_payloads = [row_payloads[i] for i in batch]
                vecs, good = embed_many(
                    batch_docs,
                    model=settings.embed_model,
                    sheet=name,
                    payloads=batch_payloads,
                )
                if len(good) != len(batch_docs) or len(vecs) != len(batch_docs):
                    raise RuntimeError(f"Incomplete index embeddings for {name}: {len(good)}/{len(batch_docs)}")
                for v, idx in zip(vecs, good):
                    original_idx = batch[idx]
                    embeddings.append(np.asarray(v, dtype=np.float32))
                    metas.append({"id": str(uuid.uuid4()), **row_payloads[original_idx]})

            log.info("Indexed sheet %s: %d rows", name, len(docs))

    if not embeddings:
        raise RuntimeError("No rows indexed. Check workbook path, sheet naming, and selection.")

    X = np.vstack(embeddings).astype("float32")
    index = FaissIPIndex(dim=X.shape[1])
    index.add(X)

    index.save(str(paths.index_path))
    pd.DataFrame(metas).to_parquet(str(paths.metas_path))
    manifest = {
        "format": "transcript-v1",
        "workbook_sha256": workbook_digest,
        "codebook_sha256": codebook_digest,
        "excluded_source_labels": sorted(excluded),
        "excluded_label_instances": excluded_instances,
        "excluded_only_rows": excluded_only_rows,
        "embed_model": settings.embed_model,
        "source_sheets": source_sheets,
        "indexed_rows": len(metas),
        "index_sha256": _file_sha256(paths.index_path),
        "metas_sha256": _file_sha256(paths.metas_path),
    }
    manifest_path = _index_manifest_path(settings)
    staged_manifest = manifest_path.with_suffix(".json.part")
    staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    staged_manifest.replace(manifest_path)

    log.info("Wrote index: %s", paths.index_path)
    log.info("Wrote metas: %s (%d rows)", paths.metas_path, len(metas))


def _load_artifacts(settings: Settings) -> tuple[FaissIPIndex, pd.DataFrame]:
    paths = artifact_paths(settings.index_artifacts_dir, settings.run_prefix)
    manifest_path = _index_manifest_path(settings)
    if not manifest_path.is_file():
        raise RuntimeError("Index manifest missing; rebuild index with the current workbook and codebook")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format": "transcript-v1",
        "workbook_sha256": _file_sha256(settings.workbook_path),
        "codebook_sha256": _file_sha256(settings.codebook_path),
        "excluded_source_labels": sorted(
            " ".join(x.casefold().split()) for x in getattr(settings, "excluded_source_labels", ())
        ),
        "embed_model": settings.embed_model,
        "index_sha256": _file_sha256(paths.index_path),
        "metas_sha256": _file_sha256(paths.metas_path),
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("Index provenance mismatch; rebuild with the active workbook/codebook/model")
    index = FaissIPIndex.load(str(paths.index_path))
    metas = pd.read_parquet(str(paths.metas_path))
    if index.index.ntotal != len(metas) or manifest.get("indexed_rows") != len(metas):
        raise ValueError("Index and metadata have different row counts")
    if sorted(set(metas["sheet"].astype(str))) != sorted(set(manifest.get("source_sheets", []))):
        raise ValueError("Index metadata sheets differ from provenance manifest")
    return index, metas


def infer_sheet(
    settings: Settings,
    *,
    team: int,
    meet: int,
    semester: str,
    embed_workers: int = 1,
    embed_batch_size: int = 32,
    lm_workers: int = 4,
    lm_timeout: float = 120.0,
    pilot_row_limit: int | None = None,
) -> None:
    """Stage C: embed → retrieve → per-label binary classify → persist raw results.

    For each transcript row:
      1. Embed the line.
      2. Retrieve top-k examples from the FAISS index.
      3. For each authoritative clean-codebook label, filter retrieved results to those
         coded with that label and build a BinaryClassificationJob.
      4. Run all jobs concurrently via classify_binary_jobs.
      5. Write raw results to artifacts/{run_prefix}raw_results.xlsx.

    Each call appends a new Run sheet (Run 0, Run 1, …) to the raw results
    workbook, enabling K-run consensus in Stage D.
    """
    if not settings.lm_api_base or not settings.lm_model:
        raise RuntimeError("LM_STUDIOS_HOST / LM_MODEL not configured")

    sheet_ref = SheetRef(team=team, meet=meet, semester=semester)
    name, df_raw = load_sheet_df(str(settings.workbook_path), sheet_ref, settings.sheet_template)
    timing_s = _timing_series(df_raw)
    df = normalize_sheet_df(df_raw)

    index, metas = _load_artifacts(settings)
    _assert_target_not_indexed(metas, name)
    metas_records = metas.to_dict(orient="records")
    codebook = load_codebook(settings.codebook_path)
    active_entries = [codebook.entry_by_label(label) for label in settings.active_labels]
    undefined = [entry.label for entry in active_entries if not entry.definition.strip()]
    if undefined:
        raise ValueError(f"Active labels lack approved definitions: {undefined}")

    top_k = settings.retrieval_top_k
    prev_k = settings.prev_k

    transcripts = df["Transcript"].astype("string").fillna("").tolist()
    speakers = df["Speaker"].astype("string").fillna("").tolist()
    work_indices = [i for i, t in enumerate(transcripts) if str(t).strip()]
    if pilot_row_limit is not None:
        if pilot_row_limit < 1:
            raise ValueError("Pilot row limit must be positive")
        if "pilot" not in settings.artifacts_dir.name.casefold():
            raise ValueError("Pilot inference requires a dedicated pilot artifacts_dir")
        work_indices = work_indices[:pilot_row_limit]
    if not work_indices:
        raise ValueError(f"No nonempty transcript rows in {name}; refusing empty run")

    # ---- Embed (with cache) ----
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.artifacts_dir / f"{settings.run_prefix}{name}_query_vectors.npz"
    qv_by_row: dict[int, np.ndarray] = {}
    embed_errors: dict[int, str] = {}

    fingerprint = _query_cache_fingerprint(transcripts, settings.embed_model)
    if cache_path.exists():
        try:
            with np.load(str(cache_path), allow_pickle=False) as loaded:
                cached_indices = loaded["indices"].tolist()
                cached_vectors = loaded["vectors"]
                cached_fingerprint = str(loaded["fingerprint"].item())
                if (cached_fingerprint == fingerprint
                        and len(cached_indices) == len(set(cached_indices))
                        and set(cached_indices) <= set(work_indices)
                        and len(cached_vectors) == len(cached_indices)
                        and cached_vectors.ndim == 2 and np.isfinite(cached_vectors).all()):
                    qv_by_row = dict(zip(cached_indices, cached_vectors))
                    log.info("Loaded cached query vectors for %s (%d rows)", name, len(qv_by_row))
                else:
                    log.warning("Query-vector cache fingerprint/shape mismatch for %s; recomputing", name)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            log.warning("Unreadable query-vector cache for %s (%s); recomputing", name, exc)

    missing_indices = [i for i in work_indices if i not in qv_by_row]
    if missing_indices:
        # Resume embedding from the last atomic batch cache.
        embed_workers = max(1, int(embed_workers))
        batches = _chunks(missing_indices, max(1, int(embed_batch_size)))

        def _save_cache() -> None:
            if not qv_by_row:
                return
            indices = np.array(sorted(qv_by_row), dtype=np.int32)
            vectors = np.array([qv_by_row[i] for i in indices], dtype=np.float32)
            staged = cache_path.with_suffix(".npz.part")
            with staged.open("wb") as stream:
                np.savez(stream, indices=indices, vectors=vectors, fingerprint=fingerprint)
            staged.replace(cache_path)

        def _embed_batch(batch: list[int]) -> dict[int, np.ndarray]:
            texts = [str(transcripts[i]).strip() for i in batch]
            vecs, good = embed_many(texts, model=settings.embed_model)
            return {batch[j]: vecs[k] for k, j in enumerate(good)}

        with tqdm(total=len(work_indices), desc="embedding", unit="row") as pbar:
            if embed_workers == 1:
                for batch in batches:
                    try:
                        qv_by_row.update(_embed_batch(batch))
                        _save_cache()
                    except Exception as e:
                        for i_int in batch:
                            embed_errors[i_int] = str(e)
                    pbar.update(len(batch))
            else:
                log.info(
                    "Embedding concurrency: embed_workers=%d embed_batch_size=%d (batches=%d)",
                    embed_workers, embed_batch_size, len(batches),
                )
                with ThreadPoolExecutor(max_workers=embed_workers) as pool:
                    futures = {pool.submit(_embed_batch, batch): batch for batch in batches}
                    for fut in as_completed(futures):
                        batch = futures[fut]
                        try:
                            qv_by_row.update(fut.result())
                            _save_cache()
                        except Exception as e:
                            for i_int in batch:
                                embed_errors[i_int] = str(e)
                        pbar.update(len(batch))

        log.info("Saved query vector cache for %s (%d rows)", name, len(qv_by_row))

    # ---- Build binary jobs ----
    jobs: list[BinaryClassificationJob] = []
    row_meta: dict[int, dict] = {}  # row_index -> {timing, speaker, transcript}
    label_order = {e.label: i for i, e in enumerate(active_entries)}
    embed_failed: list[RawResult] = []

    for i_int in work_indices:
        line = str(transcripts[i_int]).strip()
        speaker = str(speakers[i_int]).strip()
        timing = timing_s.iat[i_int]
        row_meta[i_int] = {"timing": timing, "speaker": speaker, "transcript": line}
        qv = qv_by_row.get(i_int)
        if qv is None:
            error = embed_errors.get(i_int, "embedding unavailable after retry")
            log.warning("row=%d embed error, recording failed jobs: %s", i_int, error)
            embed_failed.extend(
                RawResult(i_int, timing, speaker, line, entry.label,
                          f"__EMBED_ERROR__: {error}", "")
                for entry in active_entries
            )
            continue

        start = max(0, i_int - prev_k)
        prev_lines = [t.strip() for t in transcripts[start:i_int] if t and t.strip()][-prev_k:]

        D, I = index.search(qv, top_k)
        retrieved = _retrieved_rows(D[0], I[0], metas_records)

        for entry in active_entries:
            label_filtered = [
                (m, s) for m, s in retrieved if entry.label in m.get("codes", [])
            ]
            retrieved_examples = [_format_binary_example(m, s) for m, s in label_filtered]

            jobs.append(
                BinaryClassificationJob(
                    row_index=i_int,
                    entry=entry,
                    retrieved_examples=retrieved_examples,
                    line_to_analyze=line,
                    prev_lines=prev_lines,
                )
            )

    log.info("infer_sheet %s: %d rows, %d binary jobs, %d embed failures",
             name, len(row_meta), len(jobs), len(embed_failed) // len(active_entries))
    _classify_with_checkpoint(
        settings, name, work_indices, active_entries, jobs, embed_failed,
        row_meta, label_order, lm_workers, lm_timeout,
    )


def _classify_with_checkpoint(
    settings: Settings,
    name: str,
    work_indices: list[int],
    active_entries: list[CodebookEntry],
    jobs: list[BinaryClassificationJob],
    embed_failed: list[RawResult],
    row_meta: dict[int, dict],
    label_order: dict[str, int],
    lm_workers: int,
    lm_timeout: float,
) -> None:
    """Persist each completed batch, then export only at full per-label coverage."""
    if not settings.lm_api_base or not settings.lm_model:
        raise ValueError("Chat model and endpoint are required")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = str(settings.artifacts_dir / f"{settings.run_prefix}raw_results.xlsx")
    checkpoint_name = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    checkpoint_path = settings.artifacts_dir / f"{settings.run_prefix}{checkpoint_name}_checkpoint.sqlite3"
    fingerprint = _run_fingerprint(settings, name, lm_timeout, work_indices)
    run_sheet = next_run_sheet_name(raw_path, name)
    with CheckpointStore(checkpoint_path, fingerprint=fingerprint, run_sheet=run_sheet) as store:
        run_sheet = store.run_sheet
        completed = store.completed_keys()
        failed_pending = [r for r in embed_failed if (r.row_index, r.label) not in completed]
        if failed_pending:
            store.append(failed_pending)
            completed.update((r.row_index, r.label) for r in failed_pending)
        pending_jobs = [job for job in jobs if (job.row_index, job.label_name) not in completed]
        batch_size = max(1, len(active_entries) * 8)
        for batch in (pending_jobs[i:i + batch_size] for i in range(0, len(pending_jobs), batch_size)):
            results = asyncio.run(
                classify_binary_jobs(
                    jobs=batch,
                    api_base=settings.lm_api_base,
                    model=settings.lm_model,
                    max_concurrency=lm_workers,
                    max_retries=settings.binary_max_retries,
                    timeout_s=lm_timeout,
                    reasoning_effort=getattr(settings, "lm_reasoning_effort", None),
                    temperature=getattr(settings, "lm_temperature", 0.0),
                    top_p=getattr(settings, "lm_top_p", None),
                    max_tokens=getattr(settings, "lm_max_tokens", 10),
                    chat_template_kwargs=getattr(settings, "lm_chat_template_kwargs", None),
                    seed=getattr(settings, "lm_seed", None),
                )
            )
            expected = {(job.row_index, job.label_name) for job in batch}
            if set(results) != expected:
                raise RuntimeError(f"Classifier returned {len(results)}/{len(batch)} batch results")
            records = []
            for (row_idx, label_name), (parsed, raw_response) in results.items():
                meta = row_meta[row_idx]
                records.append(RawResult(
                    row_index=row_idx,
                    timing=meta["timing"],
                    speaker=meta["speaker"],
                    transcript=meta["transcript"],
                    label=label_name,
                    raw_response=raw_response,
                    parsed=parsed if parsed is not None else "",
                ))
            store.append(records)
        raw_records = store.all_results()
        expected_keys = {(i, entry.label) for i in work_indices for entry in active_entries}
        if {(r.row_index, r.label) for r in raw_records} != expected_keys:
            raise RuntimeError("Incomplete inference checkpoint; refusing completed Excel export")
        raw_records.sort(key=lambda r: (r.row_index, label_order.get(r.label, 999)))
        write_raw_results(raw_path, run_sheet, raw_records)
    completed_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.{run_sheet.rsplit(' ', 1)[-1]}.done.sqlite3"
    )
    checkpoint_path.replace(completed_path)
    log.info("infer_sheet %s: wrote %d records to %s [%s]", name, len(raw_records), raw_path, run_sheet)


def export_sheet(
    settings: Settings,
    *,
    team: int,
    meet: int,
    semester: str,
) -> None:
    """Stage D: read raw results → aggregate → write wide sheet.

    Reads all Run sheets for the given sheet from the raw results workbook,
    aggregates them (K=1 yes/no, K>1 confidence float), and writes a wide sheet
    to artifacts/{run_prefix}wide_results.xlsx.
    """
    sheet_ref = SheetRef(team=team, meet=meet, semester=semester)
    name = sheet_name(sheet_ref, settings.sheet_template)

    raw_path = str(settings.artifacts_dir / f"{settings.run_prefix}raw_results.xlsx")
    raw_df = read_raw_results(raw_path, name)

    if raw_df.empty:
        log.warning("export_sheet %s: no raw results found in %s", name, raw_path)
        return

    wide_df = aggregate_runs(raw_df, list(settings.active_labels))

    out_path = str(settings.artifacts_dir / f"{settings.run_prefix}wide_results.xlsx")

    import os
    if os.path.exists(out_path):
        mode, extra = "a", {"if_sheet_exists": "replace"}
    else:
        mode, extra = "w", {}

    with pd.ExcelWriter(out_path, engine="openpyxl", mode=mode, **extra) as writer:
        wide_df.to_excel(writer, sheet_name=name, index=False)

    log.info(
        "export_sheet %s: wrote %d rows to %s [sheet=%s]",
        name, len(wide_df), out_path, name,
    )
