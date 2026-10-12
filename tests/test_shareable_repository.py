"""Shareable-snapshot regressions, all fixtures synthetic."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    return tmp_path


def stage(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    subprocess.run(['git', 'add', name], cwd=root, check=True)


def test_scanner_rejects_workbook_outside_private_directories(tmp_path):
    scanner = module('check_repository')
    root = repository(tmp_path)
    stage(root, 'not-private-name.xlsx', 'synthetic bytes')
    assert not scanner.scan(root)['passed']


def test_scanner_rejects_notebook_outputs(tmp_path):
    scanner = module('check_repository')
    root = repository(tmp_path)
    stage(root, 'demo.ipynb', json.dumps({'cells': [{'cell_type': 'code', 'execution_count': 1,
                                                 'outputs': [{'text': 'synthetic'}]}]}))
    assert any(f['kind'] == 'executed_notebook' for f in scanner.scan(root)['findings'])


def test_scanner_rejects_secret_without_emitting_it(tmp_path):
    scanner = module('check_repository')
    root = repository(tmp_path)
    synthetic_secret = 'gh' + 'p_' + 'A' * 40
    stage(root, 'sample.txt', synthetic_secret)
    result = scanner.scan(root)
    assert not result['passed']
    assert synthetic_secret not in json.dumps(result)


def test_scanner_accepts_source_only(tmp_path):
    scanner = module('check_repository')
    root = repository(tmp_path)
    stage(root, 'README.md', 'Fictional documentation without private inputs.')
    assert scanner.scan(root)['passed']


def test_controlled_config_path_is_portable_and_explicit(monkeypatch, tmp_path):
    experiment = module('controlled_experiments')
    monkeypatch.delenv('PTG_STUDY_CONFIG', raising=False)
    assert experiment.study_config_path() == ROOT / 'config/study-21.yaml'
    private = tmp_path / 'study.private.yaml'
    monkeypatch.setenv('PTG_STUDY_CONFIG', str(private))
    assert experiment.study_config_path() == private


def test_watchdog_refuses_an_unconfigured_remote(monkeypatch):
    watchdog = module('vast_lease')
    monkeypatch.delenv('VAST_LEASE_CLI_MODE', raising=False)
    monkeypatch.setattr(watchdog, 'HOST', '')
    monkeypatch.setattr(watchdog.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected SSH call'))
    with pytest.raises(RuntimeError, match='VAST_LEASE_SSH_HOST'):
        watchdog._cli('show', 'instances-v1', '--raw')


def test_shareable_codebook_marks_fictional_examples():
    from ptg.classify.codebook import load_default_codebook
    book = load_default_codebook()
    assert book.version == 'shareable-synthetic-v1'
    assert all('fictional' in entry.positive_examples[0].casefold() for entry in book.entries)
