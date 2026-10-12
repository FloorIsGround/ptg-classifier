"""Run the real pipeline on fictional data with explicit network test doubles.

Good: verify a fresh clone without private data, GPUs, model downloads or services.
Bad: interpret deterministic oracle scores as measured LLM or human agreement.
All generated workbooks/indexes/checkpoints stay under ignored artifacts/.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd

from ptg.classify import binary_classifier
from ptg.config import load_settings
from ptg.pipeline import run as pipeline

ROOT = Path(__file__).resolve().parents[1]


def embeddings(texts, model, **kwargs):
    """Deterministic synthetic vectors, not an embedding-model evaluation."""
    vectors = [np.frombuffer(hashlib.sha256(str(t).encode()).digest(), dtype=np.uint8)
               .astype(np.float32) + 1 for t in texts]
    return np.vstack(vectors), list(range(len(texts)))


def demo() -> dict:
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='synthetic-demo-', dir=ROOT / 'artifacts'))
    workbook = out / 'fictional-workbook.xlsx'
    with pd.ExcelWriter(workbook) as writer:
        for team in (1, 2):
            for meet in (1, 2):
                pd.DataFrame([
                    {'Speaker': 'Fictional A', 'Transcript': f'Fictional reference {team}-{meet}: SYNTHETIC_POSITIVE[Initiative]', 'Reconciled Codes': 'Initiative'},
                    {'Speaker': 'Fictional B', 'Transcript': f'Fictional reference {team}-{meet}: SYNTHETIC_POSITIVE[Questions to group]', 'Reconciled Codes': 'Questions to group'},
                ]).to_excel(writer, sheet_name=f'Team {team} Time {meet} (F25)', index=False)
        for term, team in (('S26', 1), ('SP26', 2)):
            pd.DataFrame([
                {'Speaker': 'Fictional C', 'Transcript': f'Fictional target {term}: SYNTHETIC_POSITIVE[Initiative]', 'Reconciled Codes': 'Initiative'},
                {'Speaker': 'Fictional D', 'Transcript': f'Fictional target {term}: SYNTHETIC_POSITIVE[Questions to group]', 'Reconciled Codes': 'Questions to group'},
                {'Speaker': 'Fictional E', 'Transcript': f'Fictional neutral target for {term}; no behavior marker.', 'Reconciled Codes': ''},
            ]).to_excel(writer, sheet_name=f'Team {team} Time 1 ({term})', index=False)
    settings = replace(load_settings(ROOT / 'config/study-21.yaml'), workbook_path=workbook,
                       artifacts_dir=out / 'predictions', index_artifacts_dir=out / 'index',
                       embed_model='synthetic-hash-vector-32', lm_model='synthetic-marker-oracle',
                       lm_api_base='http://127.0.0.1:1/v1', lm_reasoning_effort=None)
    request_count = 0

    def response(request):
        nonlocal request_count
        request_count += 1
        body = json.loads(request.content)
        prompt = body['messages'][-1]['content']
        label = prompt.split('Target label: ', 1)[1].split('\n', 1)[0]
        current = prompt.split('Current line to analyze:\n', 1)[1].split('\n\nDoes the current line', 1)[0]
        result = 'yes' if f'SYNTHETIC_POSITIVE[{label}]' in current else 'no'
        return httpx.Response(200, json={'choices': [{'message': {'content': result}}]})

    original_client = httpx.AsyncClient
    with patch.object(pipeline, 'embed_many', embeddings), patch.object(
            binary_classifier.httpx, 'AsyncClient', lambda **kwargs: original_client(
                transport=httpx.MockTransport(response), **kwargs)):
        pipeline.build_index(settings, semester='F25', teams=[1, 2], meets=[1, 2])
        for term, team in (('S26', 1), ('SP26', 2)):
            pipeline.infer_sheet(settings, semester=term, team=team, meet=1, lm_workers=2)
            pipeline.export_sheet(settings, semester=term, team=team, meet=1)
    spec = importlib.util.spec_from_file_location('synthetic_demo_evaluator', ROOT / 'scripts/evaluate_reconciled.py')
    assert spec and spec.loader
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    sheets = [evaluator.score_one(settings, term, team, 1, verify_ledger=True)
              for term, team in (('S26', 1), ('SP26', 2))]
    jobs = sum(s['totals']['jobs'] for s in sheets)
    assert jobs == request_count == 126
    assert all(s['totals']['prediction_coverage'] == 1 and s['totals']['micro_f1'] == 1 for s in sheets)
    result = {'kind': 'synthetic integration test only, NOT research/model performance',
              'output_directory': str(out.relative_to(ROOT)), 'target_turns': 6,
              'labels': 21, 'decisions': jobs, 'model_network_calls': 0,
              'deterministic_oracle_f1': 1.0, 'coverage': 1.0,
              'stages_verified': ['workbook ingest', 'FAISS index and retrieval', 'binary prompts',
                                  'SQLite checkpoint', 'raw Excel', 'wide export', 'provenance-gated scoring']}
    (out / 'demo-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(demo(), indent=2, sort_keys=True))
