"""Synthetic-only experiment contracts: no private workbook or live inference."""
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/controlled_experiments.py'
spec = importlib.util.spec_from_file_location('controlled_experiments', SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def test_panel_selection_deterministic_and_gold_independent():
    texts = ['row' if i % 7 else '' for i in range(150)]
    x = exp.select_rows(texts, 'Team 1 Time 1 (S26)', 42, 64)
    assert x == exp.select_rows(texts, 'Team 1 Time 1 (S26)', 42, 64)
    assert len(x) == len(set(x)) == 64
    assert all(texts[i] for i in x)
    # Selection receives no labels or predictions; sheet and seed bind the panel.
    assert x != exp.select_rows(texts, 'Team 2 Time 1 (SP26)', 42, 64)
    assert x != exp.select_rows(texts, 'Team 1 Time 1 (S26)', 43, 64)


def test_isolated_arms_preserve_baseline_prompt():
    from ptg.classify.binary_classifier import BinaryClassificationJob, build_binary_prompt
    from ptg.domain.models import CodebookEntry
    entry = CodebookEntry(id='1', label='Label', category='cat', definition='definition',
                          positive_examples=[], negative_examples=[])
    global_rows = [({'codes':['Other'], 'speaker':'S', 'transcript':'other'}, .9),
                   ({'codes':['Label'], 'speaker':'S', 'transcript':'one'}, .8)]
    all_rows = global_rows + [({'codes':['Label'], 'speaker':'S', 'transcript':f'next{i}'}, .7-i*.01)
                              for i in range(6)]
    transcripts = [f'earlier {i}' for i in range(6)] + ['target']
    rows = exp.jobs_for_row(6, transcripts, [entry], global_rows, all_rows)
    prompts = {a:(sys,user) for a,_,sys,user in rows}
    baseline_examples = exp.select_examples('baseline','Label',global_rows,all_rows)
    assert prompts['baseline'] == build_binary_prompt(BinaryClassificationJob(
        6, entry, baseline_examples, 'target', ['earlier 4','earlier 5']))
    assert '(none found)' in prompts['no_retrieval'][1]
    assert len(exp.select_examples('label_top5','Label',global_rows,all_rows)) == 5
    assert len(baseline_examples) == 1
    assert 'earlier 0' in prompts['context_prev6'][1]
    assert 'earlier 0' not in prompts['baseline'][1]
    assert prompts['context_prev6'][0] == prompts['baseline'][0]


def test_sparse_retrieval_and_abstentions_preserve_negatives():
    assert exp.select_examples('label_top5','rare', [({'codes':['common']},.9)],
                               [({'codes':['common']},.9)]) == []
    jobs = [('baseline','S26',0,'Label',0,'system','prompt','hash'),
            ('baseline','S26',1,'Label',1,'system','prompt','hash'),
            ('baseline','S26',2,'Label',1,'system','prompt','hash')]
    responses = {('baseline','S26',0,'Label'):('yes','yes',12,1,1000000000),
                 ('baseline','S26',1,'Label'):(None,'unclear',12,1,2000000000),
                 ('baseline','S26',2,'Label'):('no','no',12,1,1000000000)}
    result = exp.score(jobs, responses, ['Label'])
    assert result['totals']['fp'] == 1
    assert result['totals']['abstained_positive'] == 1
    assert result['totals']['fn'] == 1
    assert result['totals']['support'] == 2
    assert result['micro_recall'] == 0
    assert result['prediction_coverage'] == pytest.approx(2/3)


def fake_prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, 'WORKTREE', tmp_path)
    monkeypatch.setattr(exp, 'input_hashes', lambda settings: ({'source':'original'}, {}))
    monkeypatch.setattr(exp, 'source_settings', lambda: None)
    out = tmp_path / 'artifacts' / 'unit'
    out.mkdir(parents=True)
    rows = [(arm, 'S26', 1, 'Label', 0, 'system', 'private prompt', exp.digest(['system','private prompt']))
            for arm in exp.ARMS]
    rows.sort(key=lambda x:x[:4])
    manifest = {'model':'test:4b', 'source_hashes':{'source':'original'},
                'sheets':{'S26':[1]}, 'labels':['Label'], 'seed':42, 'rows_per_sheet':1,
                'prompt_set_sha256':exp.digest(rows), 'decode':{'num_ctx':8192}}
    (out/'manifest.json').write_text(json.dumps(manifest))
    with sqlite3.connect(out/'prepared.sqlite3') as db:
        db.execute('CREATE TABLE jobs (arm TEXT, sheet TEXT, row_index INTEGER, label TEXT, gold INTEGER, system TEXT, prompt TEXT, prompt_hash TEXT, PRIMARY KEY(arm,sheet,row_index,label))')
        db.executemany('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)', rows)
    return out


def test_drift_and_incomplete_resume(tmp_path, monkeypatch):
    out = fake_prepared(tmp_path, monkeypatch)
    manifest, rows = exp.verified(out)
    assert len(rows) == 4
    monkeypatch.setattr(exp, 'input_hashes', lambda settings: ({'source':'changed'}, {}))
    with pytest.raises(ValueError, match='drift'):
        exp.verified(out)
    monkeypatch.setattr(exp, 'input_hashes', lambda settings: ({'source':'original'}, {}))
    with pytest.raises(sqlite3.OperationalError):
        exp.report(out)  # absent response ledger cannot be complete
    with sqlite3.connect(out/'prepared.sqlite3') as db:
        db.execute("UPDATE jobs SET prompt='tampered' WHERE arm='baseline'")
    with pytest.raises(ValueError, match='drift'):
        exp.verified(out)


def test_fa26_refusal_and_loopback_only(capsys):
    assert exp.SHEETS == (('S26',1,1),('SP26',2,1))
    with pytest.raises(SystemExit):
        exp.main(['prepare','--sheets','FA26:1:1'])
    with pytest.raises(ValueError, match='loopback'):
        exp.loopback('https://api.example.com')
    with pytest.raises(ValueError, match='loopback'):
        exp.loopback('http://192.168.1.113:11435')
    assert exp.loopback('http://127.0.0.1:11435') == 'http://127.0.0.1:11435'


def test_synthetic_transport_resume_and_payload(tmp_path, monkeypatch):
    out = fake_prepared(tmp_path, monkeypatch)
    calls = []
    class Reply:
        def __init__(self, body): self.body = body
        def raise_for_status(self): return self
        def json(self): return self.body
    class Client:
        def __init__(self, *args, **kwargs): assert kwargs['trust_env'] is False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json):
            assert url.endswith('/api/chat')
            calls.append(json)
            return Reply({'model':'test:4b','done':True,'message':{'content':'maybe'},
                          'prompt_eval_count':25,'eval_count':2,'total_duration':500000000})
    monkeypatch.setattr(exp.httpx, 'Client', Client)
    monkeypatch.setattr(exp, 'response_identity', lambda *args: {'model':'test:4b','model_digest':'frozen','version':'test','url':args[1]})
    result = exp.run(out,'test:4b','http://127.0.0.1:11435')
    assert result['completed'] == 4
    assert len(calls) == 4
    assert all(c['think'] is False and c['options']['num_ctx'] == 8192 and c['stream'] is False for c in calls)
    assert exp.run(out,'test:4b','http://127.0.0.1:11435')['remaining'] == 0
    assert len(calls) == 4
    summary = exp.report(out)
    assert summary['complete'] is True
    aggregate = json.loads((out/'aggregate.json').read_text())
    assert aggregate['arms']['baseline']['pooled']['totals']['abstained_negative'] == 1
    assert aggregate['arms']['baseline']['sheets']['S26']['decision_changes_vs_baseline'] == 0
    assert 'private prompt' not in (out/'aggregate.json').read_text()
    with pytest.raises(ValueError, match='fingerprint drift'):
        exp.run(out,'test:4b','http://127.0.0.1:11435',keep_alive='5m')
    with sqlite3.connect(out/'responses.sqlite3') as db:
        assert {r[0] for r in db.execute('SELECT parsed FROM responses')} == {None}
        db.execute("DELETE FROM responses WHERE arm='baseline'")
    with pytest.raises(ValueError, match='incomplete'):
        exp.report(out)
