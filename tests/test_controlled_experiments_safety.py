"""Parent regression checks for local-only controlled experiment execution."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import httpx
import pytest

spec = importlib.util.spec_from_file_location('experiment_contract_tests',
    Path(__file__).with_name('test_controlled_experiments.py'))
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
exp = base.exp


@pytest.mark.parametrize('extra, message', [
    ({'message': {'content': 'yes', 'thinking': 'unexpected'}}, 'thinking'),
    ({'done_reason': 'length'}, 'budget'),
    ({'prompt_eval_count': 8192}, 'truncated'),
])
def test_fail_closed_bounds_work_and_records_no_fake_results(tmp_path, monkeypatch, extra, message):
    out = base.fake_prepared(tmp_path, monkeypatch)
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        body = {'model': 'test:4b', 'done': True, 'message': {'content': 'yes'},
                'prompt_eval_count': 10, 'eval_count': 2, 'total_duration': 1000}
        body.update(extra)
        return httpx.Response(200, json=body)
    original_client = httpx.Client
    monkeypatch.setattr(exp.httpx, 'Client', lambda **kw: original_client(
        transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(exp, 'response_identity', lambda *args: {'model':'test:4b', 'digest':'frozen'})
    with pytest.raises(ValueError, match=message):
        exp.run(out, 'test:4b', 'http://127.0.0.1:11435')
    assert 1 <= len(calls) <= 2
    with sqlite3.connect(out/'responses.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM responses').fetchone()[0] == 0


def test_score_uses_actual_context_capacity():
    jobs = [('baseline','S26',0,'L',1,'system','prompt','hash')]
    responses = {('baseline','S26',0,'L'): ('yes','yes',6000,2,1000)}
    result = exp.score(jobs, responses, ['L'], num_ctx=8192)
    assert result['prompt_context_at_or_above_num_ctx'] == 0
    assert result['micro_f1'] == 1
    assert exp.score(jobs, responses, ['L'], num_ctx=4096)['prompt_context_at_or_above_num_ctx'] == 1


def test_failed_final_runtime_validation_blocks_report_and_resume(tmp_path, monkeypatch):
    out = base.fake_prepared(tmp_path, monkeypatch)
    original_client = httpx.Client
    def respond(request):
        return httpx.Response(200, json={'model':'test:4b', 'done':True,
            'message':{'content':'yes'}, 'prompt_eval_count':10, 'eval_count':2,
            'total_duration':1000})
    monkeypatch.setattr(exp.httpx, 'Client', lambda **kw: original_client(
        transport=httpx.MockTransport(respond), **kw))
    identities = iter([{'digest':'original'}, {'digest':'changed'}])
    monkeypatch.setattr(exp, 'response_identity', lambda *args: next(identities))
    with pytest.raises(ValueError, match='identity changed'):
        exp.run(out, 'test:4b', 'http://127.0.0.1:11435')
    with sqlite3.connect(out/'responses.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM responses').fetchone()[0] == 4
        assert not db.execute("SELECT 1 FROM metadata WHERE key='completion_validation'").fetchall()
    with pytest.raises(ValueError, match='completion validation missing'):
        exp.report(out)
    monkeypatch.setattr(exp, 'response_identity', lambda *args: {'digest':'original'})
    with pytest.raises(ValueError, match='previous runtime identity validation failed'):
        exp.run(out, 'test:4b', 'http://127.0.0.1:11435')


def test_parser_is_in_source_fingerprint(monkeypatch):
    monkeypatch.setattr(exp, '_file_sha256', lambda path: str(path))
    hashes, _ = exp.input_hashes(exp.source_settings())
    assert hashes['parser'] == str(exp.WORKTREE/'src/ptg/classify/enforce.py')
