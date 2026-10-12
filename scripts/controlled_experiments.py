"""Private, local-only paired coding experiment. No embedding or cloud calls.

Example (from this worktree; source data is read-only in the original checkout):
  python scripts/controlled_experiments.py prepare --out artifacts/controlled-01 --model qwen3.5:4b --affirm-reviewed-blanks
  python scripts/controlled_experiments.py run --out artifacts/controlled-01 --model qwen3.5:4b --url http://127.0.0.1:11435
  python scripts/controlled_experiments.py report --out artifacts/controlled-01
Only aggregate reports go to stdout. The prepared SQLite contains private prompts/row IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np

from ptg.classify import binary_classifier
from ptg.classify.binary_classifier import BinaryClassificationJob, build_binary_prompt
from ptg.classify.codebook import load_codebook
from ptg.classify.enforce import parse_yes_no
from ptg.config import load_settings
from ptg.domain.models import SheetRef
from ptg.ingest.excel import load_sheet_df, normalize_sheet_df
from ptg.pipeline.run import (_file_sha256, _format_binary_example, _load_artifacts,
                              _query_cache_fingerprint, _retrieved_rows)
from ptg.storage.files import artifact_paths
from ptg.util.codes import canonicalize_codes, parse_codes

WORKTREE = Path(__file__).resolve().parents[1]
# Defaults are portable. A preserved private study checkout can be selected explicitly.
SOURCE = Path(os.environ.get('PTG_STUDY_ROOT', WORKTREE)).resolve()
SHEETS = (("S26", 1, 1), ("SP26", 2, 1))
ARMS = ("baseline", "no_retrieval", "label_top5", "context_prev6")
DECODE = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 16}
SPECS = {
    "baseline": "original prompt; global top 5 then label filter; previous 2 lines",
    "no_retrieval": "baseline except retrieved examples empty",
    "label_top5": "baseline except search entire index then take up to 5 label-positive examples",
    "context_prev6": "baseline except previous 6 lines",
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def select_rows(transcripts: list[str], sheet: str, seed: int = 42, n: int = 64) -> list[int]:
    """Eligibility depends on nonempty text; rank depends only on seed/sheet/index."""
    eligible = [i for i, text in enumerate(transcripts) if str(text).strip()]
    if len(eligible) < n:
        raise ValueError(f"{sheet}: only {len(eligible)} nonempty rows, need {n}")
    return sorted(sorted(eligible, key=lambda i: (digest([seed, sheet, i]), i))[:n])


def select_examples(arm: str, label: str, global_rows: list[tuple[dict, float]],
                    all_rows: list[tuple[dict, float]]) -> list[str]:
    if arm == "no_retrieval":
        return []
    candidates = all_rows if arm == "label_top5" else global_rows
    selected = [(m, s) for m, s in candidates if label in m.get('codes', [])]
    if arm == 'label_top5':
        selected = selected[:5]
    return [_format_binary_example(m, s) for m, s in selected]


def jobs_for_row(i: int, transcripts: list[str], entries: list, global_rows: list,
                 all_rows: list) -> list[tuple[str, str, str, str]]:
    """Returns arm, label, system and user prompts, using the production builder."""
    line = str(transcripts[i]).strip()
    out = []
    for arm in ARMS:
        k = 6 if arm == 'context_prev6' else 2
        previous = [t.strip() for t in transcripts[max(0, i-k):i] if t and t.strip()][-k:]
        for entry in entries:
            examples = select_examples(arm, entry.label, global_rows, all_rows)
            job = BinaryClassificationJob(i, entry, examples, line, previous)
            system, user = build_binary_prompt(job)
            out.append((arm, entry.label, system, user))
    return out


def checked_out(out: Path) -> Path:
    out = out.resolve()
    root = (WORKTREE / 'artifacts').resolve()
    if out == root or root not in out.parents or out == (SOURCE / 'artifacts').resolve():
        raise ValueError("--out must be a distinct directory under this worktree's artifacts/")
    if out.is_symlink():
        raise ValueError("symlink output refused")
    return out


def study_config_path() -> Path:
    """Select a config explicitly. Good: ignored *.private.yaml; bad: old host paths."""
    path = Path(os.environ.get('PTG_STUDY_CONFIG', 'config/study-21.yaml'))
    return path if path.is_absolute() else WORKTREE / path


def source_settings():
    """Resolve offline study inputs against the selected source checkout.

    Good: set PTG_STUDY_ROOT and PTG_STUDY_CONFIG for approved private inputs.
    Bad: point a template codebook at a frozen research index with different hashes.
    """
    cfg = load_settings(study_config_path())
    return replace(cfg, workbook_path=SOURCE / cfg.workbook_path,
                   codebook_path=SOURCE / cfg.codebook_path,
                   index_artifacts_dir=SOURCE / cfg.index_artifacts_dir,
                   artifacts_dir=SOURCE / cfg.artifacts_dir, embed_model='qwen3-embedding:4b')


def input_hashes(settings):
    paths = artifact_paths(settings.index_artifacts_dir, settings.run_prefix)
    cache = {settings.sheet_template.format(team=t, meet=m, semester=s):
             settings.artifacts_dir / f'{settings.run_prefix}{settings.sheet_template.format(team=t, meet=m, semester=s)}_query_vectors.npz'
             for s, t, m in SHEETS}
    files = {'workbook': settings.workbook_path, 'codebook': settings.codebook_path,
             'index_manifest': settings.index_artifacts_dir / f'{settings.run_prefix}index_manifest.json',
             'index': paths.index_path, 'metas': paths.metas_path,
             'pipeline': WORKTREE / 'src/ptg/pipeline/run.py',
             'prompts': Path(binary_classifier.__file__), 'harness': Path(__file__),
             'parser': WORKTREE / 'src/ptg/classify/enforce.py',
             'config': study_config_path(),
             **{f'cache:{name}': path for name, path in cache.items()}}
    return {key: _file_sha256(path) for key, path in files.items()}, cache


def _atomic_json(path: Path, value: dict) -> None:
    staged = path.with_suffix(path.suffix + '.part')
    staged.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    staged.replace(path)


def prepare(out: Path, model: str, seed: int = 42, rows: int = 64,
            num_ctx: int = 8192) -> dict:
    out = checked_out(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("prepare requires an empty experiment directory; use run to resume")
    if not model.strip() or rows < 1 or num_ctx < 1024:
        raise ValueError("model, positive row count and num_ctx >= 1024 required")
    settings = source_settings()
    if len(settings.active_labels) != 21 or settings.retrieval_top_k != 5 or settings.prev_k != 2:
        raise ValueError("study-21 baseline contract changed")
    hashes, caches = input_hashes(settings)
    index, metas = _load_artifacts(settings)  # verifies index provenance against source hashes
    meta = metas.to_dict(orient='records')
    book = load_codebook(settings.codebook_path)
    entries = [book.entry_by_label(x) for x in settings.active_labels]
    if any(not e.definition.strip() for e in entries):
        raise ValueError("active label lacks approved definition")
    selected: dict[str, list[int]] = {}
    records = []
    negatives = 0
    normalize = lambda text: ' '.join(str(text).casefold().split())
    static_examples = {normalize(text) for entry in entries for text in entry.positive_examples}
    static_examples.update(normalize(ex.text) for entry in entries for ex in entry.negative_examples)
    overlaps = {}
    exposure = {arm: {'jobs': 0, 'jobs_with_examples': 0, 'total_examples': 0,
                      'max_prompt_bytes': 0} for arm in ARMS}
    for semester, team, meet in SHEETS:
        name, raw = load_sheet_df(str(settings.workbook_path), SheetRef(team, meet, semester), settings.sheet_template)
        if name in set(metas['sheet'].astype(str)) or 'Reconciled Codes' not in raw:
            raise ValueError('indexed target or missing reconciliation')
        df = normalize_sheet_df(raw)
        transcripts = df['Transcript'].astype('string').fillna('').tolist()
        # Validate ALL nonempty source codes, not just sampled rows.
        gold = {}
        excluded = {' '.join(x.casefold().split()) for x in settings.excluded_source_labels}
        for i, text in enumerate(transcripts):
            if not str(text).strip():
                continue
            codes = [c for c in parse_codes(df.at[i, 'Reconciled Codes'])
                     if ' '.join(c.casefold().split()) not in excluded]
            gold[i] = set(canonicalize_codes(codes, codebook=book))
        if not any(gold.values()):
            raise ValueError(f'{name}: no evidence of reconciliation')
        indices = select_rows(transcripts, name, seed, rows)
        selected[name] = indices
        overlaps[name] = [i for i in indices if normalize(transcripts[i]) in static_examples]
        negatives += sum(not gold[i] for i in indices)
        cache_path = caches[name]
        with np.load(cache_path, allow_pickle=False) as loaded:
            if str(loaded['fingerprint'].item()) != _query_cache_fingerprint(transcripts, settings.embed_model):
                raise ValueError(f'{name}: query cache fingerprint mismatch')
            cached_indices = [int(i) for i in loaded['indices']]
            vectors = loaded['vectors']
            if (len(cached_indices) != len(set(cached_indices)) or len(vectors) != len(cached_indices)
                or vectors.ndim != 2 or vectors.shape[1] != index.index.d
                or not np.isfinite(vectors).all()):
                raise ValueError(f'{name}: malformed query cache')
            vector_map = dict(zip(cached_indices, vectors))
            if not set(indices) <= vector_map.keys():
                raise ValueError(f'{name}: selected rows lack cached vectors; no embedding permitted')
        for i in indices:
            # FAISS flat IP gives global similarity order. Re-use identical results
            # for the baseline and intervention; no new embedding/search API calls.
            distances, ids = index.search(vector_map[i], index.index.ntotal)
            all_rows = _retrieved_rows(distances[0], ids[0], meta)
            # Preserve the baseline's actual top-5 call, including FAISS tie ordering.
            global_distances, global_ids = index.search(vector_map[i], 5)
            global_rows = _retrieved_rows(global_distances[0], global_ids[0], meta)
            for arm, label, system, user in jobs_for_row(i, transcripts, entries, global_rows, all_rows):
                # Conservative byte-level upper bound for textual tokens plus
                # chat-template/role and response overhead. No private prompt is
                # sent to a tokenizer service during preparation.
                if len((system + user).encode('utf-8')) + 1024 + DECODE['num_predict'] >= num_ctx:
                    raise ValueError(f'{name}: prompt exceeds conservative context bound; increase --num-ctx')
                example_count = len(select_examples(arm, label, global_rows, all_rows))
                exposure[arm]['jobs'] += 1
                exposure[arm]['jobs_with_examples'] += int(example_count > 0)
                exposure[arm]['total_examples'] += example_count
                exposure[arm]['max_prompt_bytes'] = max(exposure[arm]['max_prompt_bytes'],
                                                       len((system + user).encode('utf-8')))
                records.append((arm, name, i, label, int(label in gold[i]), system, user,
                                digest([system, user])))
    records.sort(key=lambda j: j[:4])
    decode = {**DECODE, 'num_ctx': num_ctx}
    manifest = {'schema': 1, 'model': model, 'seed': seed, 'rows_per_sheet': rows,
                'sheets': selected, 'labels': list(settings.active_labels), 'arms': SPECS,
                'decode': decode, 'think': False, 'source_hashes': hashes,
                'reviewed_blanks_attested': True,
                'static_example_overlap_rows': overlaps, 'retrieval_exposure': exposure,
                'prompt_set_sha256': digest(records)}
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    db = out / 'prepared.sqlite3'
    try:
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE jobs (arm TEXT, sheet TEXT, row_index INTEGER, label TEXT, gold INTEGER, system TEXT, prompt TEXT, prompt_hash TEXT, PRIMARY KEY(arm,sheet,row_index,label))')
            conn.executemany('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)', records)
        os.chmod(db, 0o600)
        _atomic_json(out / 'manifest.json', manifest)
        os.chmod(out / 'manifest.json', 0o600)
    except Exception:
        db.unlink(missing_ok=True)
        raise
    return {'prepared': str(out), 'sheets': len(selected), 'rows': sum(map(len, selected.values())),
            'reviewed_no_code_rows': negatives, 'jobs': len(records)}


def verified(out: Path) -> tuple[dict, list[tuple]]:
    out = checked_out(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    settings = source_settings()
    hashes, _ = input_hashes(settings)
    if hashes != manifest['source_hashes']:
        raise ValueError('input/code drift: source fingerprint differs; refuse resume/report')
    with sqlite3.connect(f"{(out / 'prepared.sqlite3').as_uri()}?mode=ro", uri=True) as db:
        records = db.execute('SELECT arm,sheet,row_index,label,gold,system,prompt,prompt_hash FROM jobs ORDER BY arm,sheet,row_index,label').fetchall()
    if digest(records) != manifest['prompt_set_sha256']:
        raise ValueError('prepared prompts/gold drift')
    expected = {(arm, sheet, row, label) for arm in ARMS for sheet, rows in manifest['sheets'].items()
                for row in rows for label in manifest['labels']}
    if (len(records) != len(expected) or {(a,s,i,l) for a,s,i,l,*_ in records} != expected
        or any(digest([system, prompt]) != ph for _,_,_,_,_,system,prompt,ph in records)):
        raise ValueError('missing, duplicate or modified prepared jobs')
    return manifest, records


def loopback(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != 'http' or not parts.hostname or not parts.port or parts.username
        or parts.password or parts.query or parts.fragment or parts.path not in ('', '/')):
        raise ValueError('require bare HTTP loopback Ollama origin with explicit port')
    try:
        if not ipaddress.ip_address(parts.hostname).is_loopback:
            raise ValueError('only numeric loopback addresses permitted')
    except ValueError as exc:
        raise ValueError('only numeric loopback addresses permitted') from exc
    return url.rstrip('/')


def response_identity(client: httpx.Client, url: str, model: str) -> dict:
    version = client.get(url + '/api/version').raise_for_status().json()['version']
    shown = client.post(url + '/api/show', json={'model': model}).raise_for_status().json()
    # Digest from installed model listing, not mutable display name alone.
    models = client.get(url + '/api/tags').raise_for_status().json()['models']
    matches = [m for m in models if m.get('name') == model or m.get('model') == model]
    if len(matches) != 1 or not matches[0].get('digest'):
        raise ValueError('model not found exactly once with an Ollama digest')
    return {'url': url, 'version': version, 'model': model,
            'model_digest': matches[0]['digest'], 'details': shown.get('details', {})}


def run(out: Path, model: str, url: str, workers: int = 2, keep_alive: str = '10m',
        timeout: float = 120.0) -> dict:
    manifest, jobs = verified(out)
    if model != manifest['model'] or workers not in (2, 4) or timeout <= 0:
        raise ValueError('model mismatch, concurrency must be 2/4, or invalid timeout')
    url = loopback(url)
    if not keep_alive or len(keep_alive) > 32:
        raise ValueError('invalid keep_alive')
    out = checked_out(out)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        identity = response_identity(client, url, model)
    fingerprint = digest({'manifest': manifest, 'runtime': identity,
                          'keep_alive': keep_alive, 'workers': workers})
    db_path = out / 'responses.sqlite3'
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS responses (arm TEXT, sheet TEXT, row_index INTEGER, label TEXT, prompt_hash TEXT, parsed TEXT, raw TEXT, prompt_eval_count INTEGER, eval_count INTEGER, total_duration_ns INTEGER, PRIMARY KEY(arm,sheet,row_index,label))')
        previous = dict(db.execute('SELECT key,value FROM metadata'))
        if previous.get('runtime_validation_failed'):
            raise ValueError('previous runtime identity validation failed; prepare a new experiment')
        if previous and (previous.get('fingerprint') != fingerprint or
                         previous.get('keep_alive') != keep_alive):
            raise ValueError('runtime/model/input fingerprint drift; refusing resume')
        if not previous:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('fingerprint', fingerprint))
            db.execute('INSERT INTO metadata VALUES (?,?)', ('runtime', json.dumps(identity, sort_keys=True)))
            db.execute('INSERT INTO metadata VALUES (?,?)', ('keep_alive', keep_alive))
            db.execute('INSERT INTO metadata VALUES (?,?)', ('workers', str(workers)))
            db.commit()
        done = {(a,s,i,l): ph for a,s,i,l,ph in db.execute('SELECT arm,sheet,row_index,label,prompt_hash FROM responses')}
        expected = {(a,s,i,l): ph for a,s,i,l,_,_,_,ph in jobs}
        if any(key not in expected or expected[key] != ph for key, ph in done.items()):
            raise ValueError('stale or extra response keys')
        # Any invocation must revalidate the final runtime identity before reporting.
        db.execute("DELETE FROM metadata WHERE key='completion_validation'")
        db.commit()
        pending = sorted([job for job in jobs if job[:4] not in done],
                         key=lambda job: digest([manifest['seed'], *job[:4]]))
        os.chmod(db_path, 0o600)
        started = time.monotonic()
        def one(job):
            arm, sheet, i, label, gold, system, prompt, ph = job
            payload = {'model': model, 'messages': [{'role':'system','content':system},
                       {'role':'user','content':prompt}], 'stream': False, 'think': False,
                       'options': manifest['decode'], 'keep_alive': keep_alive}
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                data = client.post(url + '/api/chat', json=payload).raise_for_status().json()
            if data.get('model') != model or not data.get('done', False):
                raise ValueError('unexpected model or incomplete Ollama reply')
            if (data.get('prompt_eval_count') is not None and
                data['prompt_eval_count'] >= manifest['decode']['num_ctx']):
                raise ValueError('Ollama prompt may be truncated; stop and enlarge context')
            if data['message'].get('thinking'):
                raise ValueError('unexpected thinking output in non-thinking arm')
            if data.get('done_reason') == 'length':
                raise ValueError('output budget exhausted; no valid completed decision')
            raw = data['message'].get('content') or ''
            return (arm, sheet, i, label, ph, parse_yes_no(raw), raw,
                    data.get('prompt_eval_count'), data.get('eval_count'), data.get('total_duration'))
        # HTTP/JSON errors stop without storing an invented abstention; already
        # committed calls remain resumable. Invalid model content IS an abstention.
        pool = ThreadPoolExecutor(max_workers=workers)
        iterator = iter(pending)
        futures = {pool.submit(one, job) for job in [next(iterator, None) for _ in range(workers)]
                   if job is not None}
        completed_now = 0
        try:
            while futures:
                finished, futures = wait(futures, return_when=FIRST_COMPLETED)
                # Validate all just-finished replies before scheduling more work.
                batch = [future.result() for future in finished]
                db.executemany('INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?)', batch)
                db.commit()
                completed_now += len(batch)
                if completed_now % 128 < len(batch):
                    print(json.dumps({'completed': len(done)+completed_now, 'expected': len(jobs),
                                      'elapsed_seconds': round(time.monotonic()-started, 1)}), flush=True)
                for _ in batch:
                    job = next(iterator, None)
                    if job is not None:
                        futures.add(pool.submit(one, job))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            if response_identity(client, url, model) != identity:
                db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                           ('runtime_validation_failed', 'true'))
                db.commit()
                raise ValueError('runtime/model identity changed during inference')
        count = db.execute('SELECT count(*) FROM responses').fetchone()[0]
        if count != len(jobs):
            raise ValueError('incomplete response ledger after inference')
        db.execute('INSERT INTO metadata VALUES (?,?)',
                   ('completion_validation', digest({'fingerprint': fingerprint, 'count': count})))
        db.commit()
    os.chmod(db_path, 0o600)
    return {'completed': count, 'expected': len(jobs), 'remaining': len(jobs)-count,
            'response_db': str(db_path)}


def score(jobs: list[tuple], responses: dict, labels: list[str], num_ctx: int = 8192) -> dict:
    cells = {label: dict.fromkeys(('tp','fp','fn','tn','abstained_positive','abstained_negative'), 0)
             for label in labels}
    latencies = []
    prompt_counts = []
    for a,s,i,label,gold,system,prompt,ph in jobs:
        response = responses.get((a,s,i,label))
        if response is None:
            continue
        parsed, _, prompt_n, _, duration = response
        key = ('abstained_positive' if gold else 'abstained_negative') if parsed not in ('yes','no') else (
              'tp' if gold else 'fp') if parsed == 'yes' else ('fn' if gold else 'tn')
        cells[label][key] += 1
        if duration is not None:
            latencies.append(duration / 1e9)
        if prompt_n is not None:
            prompt_counts.append(prompt_n)
    def metrics(c):
        tp, fp, fn, ap = (c[x] for x in ('tp','fp','fn','abstained_positive'))
        support = tp + fn + ap
        return {**c, 'support': support,
                'precision': tp/(tp+fp) if tp+fp else None,
                'recall': tp/support if support else None,
                'f1': 2*tp/(2*tp+fp+fn+ap) if support else None}
    by_label = {l:metrics(c) for l,c in cells.items()}
    totals = {k: sum(c[k] for c in cells.values()) for k in next(iter(cells.values()))}
    overall = metrics(totals)
    supported = [c['f1'] for c in by_label.values() if c['support']]
    evaluated = sum(totals.values())
    return {'micro_precision': overall['precision'], 'micro_recall': overall['recall'],
            'micro_f1': overall['f1'], 'macro_f1_supported_labels': statistics.mean(supported) if supported else None,
            'supported_label_count': len(supported), 'prediction_coverage':
            (evaluated-totals['abstained_positive']-totals['abstained_negative'])/evaluated if evaluated else None,
            'latency_seconds_mean': statistics.mean(latencies) if latencies else None,
            'prompt_eval_count_max': max(prompt_counts) if prompt_counts else None,
            'prompt_context_at_or_above_num_ctx': sum(n >= num_ctx for n in prompt_counts),
            'prompt_usage_observed': len(prompt_counts), 'jobs_scored': evaluated,
            'totals': overall, 'labels': by_label}


def report(out: Path) -> dict:
    manifest, jobs = verified(out)
    out = checked_out(out)
    with sqlite3.connect(f"{(out/'responses.sqlite3').as_uri()}?mode=ro", uri=True) as db:
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        responses = {(a,s,i,l): (p, raw, pc, ec, dur)
                     for a,s,i,l,ph,p,raw,pc,ec,dur in db.execute('SELECT * FROM responses')}
        response_hashes = {(a,s,i,l): ph for a,s,i,l,ph in db.execute('SELECT arm,sheet,row_index,label,prompt_hash FROM responses')}
    if metadata.get('fingerprint') != digest({'manifest': manifest,
            'runtime': json.loads(metadata['runtime']), 'keep_alive': metadata.get('keep_alive', '10m'),
            'workers': int(metadata['workers'])}):
        # Legacy-free manifest: keep_alive is stored in metadata by run below.
        raise ValueError('response metadata fingerprint mismatch')
    expected = {(a,s,i,l): ph for a,s,i,l,_,_,_,ph in jobs}
    if any(key not in expected or expected[key] != ph for key,ph in response_hashes.items()):
        raise ValueError('stale/extra responses')
    if any(r[2] is not None and r[2] >= manifest['decode']['num_ctx'] for r in responses.values()):
        raise ValueError('possible prompt truncation; report refused')
    missing = set(expected)-set(responses)
    if missing:
        raise ValueError(f'incomplete experiment: {len(missing)} missing decisions; report refused')
    expected_validation = digest({'fingerprint': metadata['fingerprint'], 'count': len(jobs)})
    if metadata.get('completion_validation') != expected_validation:
        raise ValueError('completion validation missing: final runtime check was not successful')
    results = {}
    for arm in ARMS:
        arm_jobs = [j for j in jobs if j[0] == arm]
        sheets = {name: score([j for j in arm_jobs if j[1] == name], responses, manifest['labels'], manifest['decode']['num_ctx'])
                  for name in manifest['sheets']}
        results[arm] = {'pooled': score(arm_jobs, responses, manifest['labels'], manifest['decode']['num_ctx']), 'sheets': sheets}
        for name in sheets:
            sheets[name]['decision_changes_vs_baseline'] = sum(
                responses[j[:4]][0] != responses[('baseline', *j[1:4])][0]
                for j in arm_jobs if j[1] == name)
        results[arm]['decision_changes_vs_baseline'] = (0 if arm == 'baseline' else sum(
            responses[j[:4]][0] != responses[('baseline', *j[1:4])][0] for j in arm_jobs))
    overlap_keys = {(sheet, row) for sheet, rows in manifest.get('static_example_overlap_rows', {}).items()
                    for row in rows}
    sensitivity = {arm: score([j for j in jobs if j[0] == arm and j[1:3] not in overlap_keys],
                               responses, manifest['labels'], manifest['decode']['num_ctx'])
                   for arm in ARMS}
    # Paired row bootstrap: resample rows within each meeting, keeping all labels
    # and all arm decisions together. Descriptive interval, not generalization.
    rng = np.random.default_rng(manifest['seed'])
    by_row = {(s,i): [j for j in jobs if j[1:3] == (s,i)]
              for s, indices in manifest['sheets'].items() for i in indices}
    intervals = {}
    for arm in ARMS[1:]:
        differences = []
        for _ in range(300):
            sampled = [j for s,indices in manifest['sheets'].items()
                       for i in rng.choice(indices, len(indices), replace=True)
                       for j in by_row[(s,int(i))]]
            b = score([j for j in sampled if j[0]=='baseline'], responses, manifest['labels'])['micro_f1']
            x = score([j for j in sampled if j[0]==arm], responses, manifest['labels'])['micro_f1']
            if b is not None and x is not None:
                differences.append(x-b)
        intervals[arm] = [float(x) for x in np.quantile(differences, [0.025, 0.975])] if differences else None
    caveats = ['Local Qwen screening is not a Granite 4.2 gain.',
               'The same two development meetings and selected panel are not held-out evaluation.',
               'Label-top5 changes retrieved example availability and count, not just relevance.',
               'Paired row bootstrap is descriptive only: rows within two meetings are not independent populations.',
               'Prompt usage >= num_ctx flags possible truncation; absence of a flag cannot prove no truncation.']
    payload = {'complete': True, 'model': manifest['model'], 'runtime': json.loads(metadata['runtime']),
               'decode': manifest['decode'], 'arms': results, 'bootstrap_micro_f1_difference_95pct': intervals,
               'caveats': caveats, 'panel_rows_per_sheet': manifest['rows_per_sheet'],
               'source_hashes': manifest['source_hashes'],
               'retrieval_exposure': manifest.get('retrieval_exposure', {}),
               'static_example_overlap_excluded_rows': len(overlap_keys),
               'sensitivity_without_static_example_overlap': sensitivity}
    _atomic_json(out / 'aggregate.json', payload)
    lines = ['# Controlled coding experiment', '', 'Complete paired development panel; private text/IDs omitted.',
             '', '| Arm | Pooled micro P | R | F1 | Macro supported F1 | Coverage | Changes vs baseline |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        r = results[arm]['pooled']
        fmt = lambda v: 'n/a' if v is None else f'{v:.4f}'
        lines.append(f"| {arm} | {fmt(r['micro_precision'])} | {fmt(r['micro_recall'])} | {fmt(r['micro_f1'])} | {fmt(r['macro_f1_supported_labels'])} | {fmt(r['prediction_coverage'])} | {results[arm]['decision_changes_vs_baseline']} |")
    lines += ['', 'Per-sheet and per-label confusion counts, latency and bootstrap intervals: `aggregate.json`.',
              '', '## Caveats', *['- '+x for x in caveats], '']
    (out/'aggregate.md').write_text('\n'.join(lines), encoding='utf-8')
    return {'complete': True, 'json': str(out/'aggregate.json'), 'markdown': str(out/'aggregate.md')}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare', help='freeze private paired prompts from two reconciled sheets')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--num-ctx', type=int, default=8192)
    p.add_argument('--affirm-reviewed-blanks', action='store_true', required=True)
    p = sub.add_parser('run', help='local Ollama inference, resumable')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--url', required=True, help='bare numeric loopback origin, e.g. http://127.0.0.1:11435')
    p.add_argument('--workers', type=int, choices=(2,4), default=2)
    p.add_argument('--keep-alive', default='10m')
    p.add_argument('--timeout', type=float, default=120)
    p = sub.add_parser('report', help='refuses incomplete coverage; writes aggregate-only JSON/MD')
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        result = prepare(args.out, args.model, args.seed, args.rows, args.num_ctx)
    elif args.phase == 'run':
        result = run(args.out, args.model, args.url, args.workers, args.keep_alive, args.timeout)
    else:
        result = report(args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
