"""Strict v2 contract and canonical identity. No private data is exportable here."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'classifier-suite/2.0'
D1_REVISION = '051bcc464b01b9f92942b364d9586b0ef5912432'
GRANITE_REVISION = 'f8de16cdcdbc6c779ca517604e050d82cc119e44'


class ContractError(ValueError):
    pass


class AdmissionBlocked(ContractError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(ok, message):
    if not ok:
        raise ContractError(message)


def keys(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected.split()),
            'schema keys differ')


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer')


def text(value):
    require(isinstance(value, str) and bool(value.strip()), 'nonempty string required')


def labels_valid(labels):
    require(isinstance(labels, list) and len(labels) == 21 and
            all(isinstance(x, str) and x.strip() for x in labels) and len(set(labels)) == 21,
            'exactly 21 distinct ordered binary labels required')


def recipe_valid(r):
    keys(r, 'schema name backend model revision packing context retrieval decoding retry threshold')
    require(r['schema'] == VERSION, 'unsupported schema')
    require(r['name'] in recipe_names(), 'unknown recipe name')
    require(r['backend'] in ('ollama', 'granite_remote', 'd1_native'), 'unknown backend')
    text(r['model']); text(r['revision'])
    require(r['packing'] in ('single_label', 'shared_state'), 'invalid packing')
    keys(r['context'], 'policy predecessors budget_bytes decay')
    c = r['context']
    require(c['policy'] in ('fixed_predecessors', 'recency_decay_budget'), 'invalid context policy')
    integer(c['predecessors']); integer(c['budget_bytes'], 1)
    require(c['decay'] == '1/2', 'only exact rational decay 1/2 supported')
    keys(r['retrieval'], 'policy top_k')
    require(r['retrieval']['policy'] in ('frozen_top_k_label_filter', 'none'), 'invalid retrieval')
    integer(r['retrieval']['top_k'], 1)
    keys(r['decoding'], 'temperature top_p seed num_ctx num_predict think')
    d = r['decoding']
    require(type(d['temperature']) in (float, int) and math.isfinite(d['temperature']) and
            d['temperature'] >= 0, 'invalid temperature')
    require(type(d['top_p']) in (int, float) and 0 < d['top_p'] <= 1, 'invalid top_p')
    integer(d['seed']); integer(d['num_ctx'], 1); integer(d['num_predict'])
    require(d['think'] is False, 'thinking disabled')
    keys(r['retry'], 'max_attempts retry_on')
    integer(r['retry']['max_attempts'], 1)
    require(r['retry']['retry_on'] == ['transport'], 'only transport retries supported')
    require(r['threshold'] == {'value': '0.5', 'comparator': '>='}, 'fixed threshold required')
    if r['backend'] == 'd1_native':
        require(r['model'] == 'LiquidAI/d1-3B' and r['revision'] == D1_REVISION and
                d['num_predict'] == 0 and r['retrieval']['policy'] == 'none', 'invalid D1 contract')
    elif r['backend'] == 'granite_remote':
        require(r['model'] == 'ibm-granite/granite-4.2-8b' and
                r['revision'] == GRANITE_REVISION, 'invalid Granite pin')
    else:
        require(r['packing'] == 'single_label', 'Ollama single label only')
    return r


def recipe_names():
    return ['rag_baseline', 'no_retrieval', 'context_prev0', 'context_prev2',
            'context_prev6', 'context_recency_decay_budget', 'granite42_confirmation',
            'd1_single_label', 'd1_shared_state']


def recipes():
    value = json.loads((ROOT / 'config/experiments/suite-v2.json').read_text())
    require(isinstance(value, list) and [r['name'] for r in value] == recipe_names(),
            'recipe registry drift')
    return {r['name']: recipe_valid(r) for r in value}


def runtime_valid(value):
    keys(value, 'backend model artifact_digest engine_version show_digest quantization '
                'token_bound template_overhead_tokens hardware_identity')
    for k in ('backend', 'model', 'engine_version', 'quantization', 'hardware_identity'):
        text(value[k])
    for k in ('artifact_digest', 'show_digest'):
        require(isinstance(value[k], str) and len(value[k]) == 64 and
                all(c in '0123456789abcdef' for c in value[k]), 'SHA256 required')
    require(value['token_bound'] == 'utf8_bytes_upper_bound_v1', 'unvetted tokenizer bound')
    integer(value['template_overhead_tokens'], 1)
    return value


def implementation_identity():
    # Explicit transitive source allowlist; never discover private files with a glob.
    files = ['src/ptg/experiments/' + n + '.py' for n in
             ('__init__', 'contract', 'policy', 'adapters', 'suite', 'analysis', 'cli')]
    files += ['scripts/classifier_suite_v2.py', 'config/experiments/suite-v2.json',
              'pyproject.toml', 'uv.lock']
    return {'version': VERSION, 'files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                       for p in files},
            'python': sys.version, 'dependencies': dict(sorted(
                (d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}
