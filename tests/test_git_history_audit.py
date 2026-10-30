"""History privacy checks use only fictional local Git repositories."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scanner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('audit_git_history', ROOT / 'scripts/audit_git_history.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    return tmp_path


def commit(repo, message='Synthetic commit'):
    subprocess.run(['git', 'add', '-A'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=Fictional Tester', '-c', 'user.email=test@example.invalid',
                    'commit', '-q', '-m', message], cwd=repo, check=True)


def test_deleted_secret_is_detected_without_disclosure(repo, scanner):
    secret = 'gh' + 'p_' + 'A' * 40
    (repo / 'notes.txt').write_text(secret)
    commit(repo)
    (repo / 'notes.txt').write_text('Nothing sensitive in HEAD.')
    commit(repo)
    report = scanner.audit(repo)
    assert not report['passed']
    assert report['commits_checked'] == 2
    assert any(f['kind'] == 'secret_pattern' for f in report['findings'])
    assert secret not in json.dumps(report)


def test_deleted_workbook_path_and_disguised_binary_are_detected(repo, scanner):
    (repo / 'removed.xlsx').write_bytes(b'PK\x03\x04fictional')
    (repo / 'looks-like-source.txt').write_bytes(b'SQLite format 3\x00fictional')
    commit(repo)
    (repo / 'removed.xlsx').unlink()
    (repo / 'looks-like-source.txt').write_text('Clean replacement')
    commit(repo)
    kinds = {f['kind'] for f in scanner.audit(repo)['findings']}
    assert {'protected_path', 'embedded_protected_binary'} <= kinds


def test_reference_phrase_in_deleted_json_is_detected(repo, scanner):
    phrase = 'Fictional observatory volunteers carefully measured unusual planetary shadows before dinner tonight'
    (repo / 'message.json').write_text(json.dumps({'text': phrase}, ensure_ascii=True))
    commit(repo)
    (repo / 'message.json').unlink()
    (repo / 'README.md').write_text('Fictional public example')
    commit(repo)
    phrase_tokens = tuple(scanner.tokens(phrase))
    report = scanner.audit(repo, phrases={phrase_tokens}, words=len(phrase_tokens))
    assert not report['passed']
    assert any(f['kind'] == 'private_reference_text' for f in report['findings'])
    assert phrase not in json.dumps(report)


def test_commit_messages_are_scanned(repo, scanner):
    (repo / 'README.md').write_text('A public source-only repository')
    secret = 'sk-' + 'Z' * 40
    commit(repo, 'Fictional leaked token ' + secret)
    report = scanner.audit(repo)
    assert not report['passed']
    assert any(f['paths'] == ['<commit-message>'] for f in report['findings'])
    assert secret not in json.dumps(report)


def test_source_only_history_passes_and_ref_options_are_rejected(repo, scanner):
    (repo / 'README.md').write_text('Fictional public source and synthetic examples only.')
    commit(repo)
    report = scanner.audit(repo, refs=['HEAD'])
    assert report['passed']
    assert report['unique_blobs_checked'] == 1
    with pytest.raises(ValueError, match='explicit Git ref'):
        scanner.audit(repo, refs=['--not'])


def test_private_reference_loader_uses_transcript_columns_and_examples(tmp_path, scanner):
    import openpyxl
    book = openpyxl.Workbook()
    assert book.active is not None
    book.active.append(['Transcript', 'Other'])
    phrase = 'Fictional astronomers carefully prepared transparent demonstration instruments for observing distant planets'
    book.active.append([phrase, 'This other cell should never contribute protected transcript windows'])
    path = tmp_path / 'fictional.xlsx'
    book.save(path)
    codebook = tmp_path / 'fictional.yaml'
    codebook.write_text('labels:\n- name: Example\n  positive_examples:\n  - Fictional researchers reliably assembled cardboard instruments while discussing distant galaxies tonight\n')
    windows = scanner.private_phrases([path], [codebook], 10)
    assert tuple(scanner.tokens(phrase)[:10]) in windows
    assert tuple(scanner.tokens('This other cell should never contribute protected transcript windows')) not in windows
    assert len(windows) >= 2


def test_empty_history_cannot_pass(repo, scanner):
    with pytest.raises(ValueError, match='no reachable commits'):
        scanner.audit(repo)


def test_deleted_symlinks_require_review(repo, scanner):
    (repo / 'old-link').symlink_to('/fictional/private/source.txt')
    commit(repo)
    (repo / 'old-link').unlink()
    (repo / 'README.md').write_text('Clean current source.')
    commit(repo)
    report = scanner.audit(repo)
    assert not report['passed']
    assert any(f['kind'] == 'symlink_requires_review' for f in report['findings'])
