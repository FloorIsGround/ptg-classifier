"""Audit retrieval availability without inference or transcript output.

Good: audit the frozen F25 index against the two reconciled development sheets.
Bad: interpret example availability as classification accuracy or change the index.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ptg.classify.codebook import load_codebook
from ptg.config import load_settings
from ptg.domain.models import SheetRef
from ptg.ingest.excel import load_sheet_df, normalize_sheet_df
from ptg.pipeline.run import _load_artifacts, _query_cache_fingerprint, _file_sha256
from ptg.util.codes import canonicalize_codes, parse_codes


def audit(root: Path) -> dict:
    root = root.resolve()
    settings = load_settings(root / 'config/study-21-granite-4.2.yaml')
    settings = replace(settings, workbook_path=root / settings.workbook_path,
                       codebook_path=root / settings.codebook_path,
                       index_artifacts_dir=root / settings.index_artifacts_dir,
                       artifacts_dir=root / settings.artifacts_dir,
                       embed_model='qwen3-embedding:4b')
    index, metas = _load_artifacts(settings)
    book = load_codebook(settings.codebook_path)
    codes = [set(c) for c in metas['codes']]
    counts = {label: {'indexed_positive_examples': sum(label in c for c in codes),
                      'target_jobs': 0, 'jobs_with_global_top5_example': 0,
                      'human_positive_jobs': 0, 'human_positive_jobs_with_example': 0}
              for label in settings.active_labels}
    exclusions = {' '.join(c.casefold().split()) for c in settings.excluded_source_labels}
    rows_total = 0
    cache_hashes = {}
    for semester, team in [('S26', 1), ('SP26', 2)]:
        name, df = load_sheet_df(str(settings.workbook_path),
                                SheetRef(team=team, meet=1, semester=semester),
                                settings.sheet_template)
        if name in set(metas['sheet']):
            raise ValueError('Evaluation sheet is present in retrieval corpus')
        df = normalize_sheet_df(df)
        texts = df['Transcript'].astype('string').fillna('').tolist()
        eligible = {i for i, t in enumerate(texts) if str(t).strip()}
        cache = settings.artifacts_dir / f'{settings.run_prefix}{name}_query_vectors.npz'
        cache_hashes[name] = _file_sha256(cache)
        with np.load(cache, allow_pickle=False) as data:
            ids = data['indices'].tolist()
            vectors = data['vectors']
            if (str(data['fingerprint'].item()) != _query_cache_fingerprint(texts, settings.embed_model)
                    or set(ids) != eligible or len(ids) != len(eligible)
                    or len(vectors) != len(ids) or not np.isfinite(vectors).all()):
                raise ValueError('Query cache provenance or coverage mismatch')
            for i, vector in zip(ids, vectors):
                rows_total += 1
                _, neighbors = index.search(vector, 5)
                available = set().union(*(codes[int(j)] for j in neighbors[0] if 0 <= j < len(codes)))
                retained = [c for c in parse_codes(df.at[i, 'Reconciled Codes'])
                            if ' '.join(c.casefold().split()) not in exclusions]
                gold = set(canonicalize_codes(retained, codebook=book))
                for label in settings.active_labels:
                    c = counts[label]
                    c['target_jobs'] += 1
                    c['jobs_with_global_top5_example'] += int(label in available)
                    c['human_positive_jobs'] += int(label in gold)
                    c['human_positive_jobs_with_example'] += int(label in gold and label in available)
    return {'scope': 'full S26/SP26 development; availability only, not model quality',
            'rows': rows_total, 'labels': len(counts),
            'jobs': sum(c['target_jobs'] for c in counts.values()),
            'jobs_with_global_top5_example': sum(c['jobs_with_global_top5_example'] for c in counts.values()),
            'human_positive_jobs': sum(c['human_positive_jobs'] for c in counts.values()),
            'human_positive_jobs_with_example': sum(c['human_positive_jobs_with_example'] for c in counts.values()),
            'query_cache_sha256': cache_hashes,
            'workbook_sha256': _file_sha256(settings.workbook_path), 'per_label': counts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.source_root.resolve()):
        parser.error('Output cannot be written inside preserved source root')
    result = audit(args.source_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
