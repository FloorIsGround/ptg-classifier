"""Check tracked source and history for common private-data hazards.

Good: run before publication, optionally with a PRIVATE local reference workbook.
Bad: treat a passing exact-match scan as proof that paraphrases or PII are absent.
Only paths/counts are emitted, never matching transcript text or credentials.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.xlsx', '.xls', '.parquet', '.faiss', '.npz', '.sqlite',
                      '.sqlite3', '.db', '.pdf', '.docx', '.zip', '.pkl', '.pickle'}
PRIVATE_DIRS = {'data', 'artifacts', 'private', 'research-inputs', '.venv', '__pycache__'}
SECRET_PATTERNS = (
    re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),
    re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
)


def prohibited(path: str) -> bool:
    """Good: reject a workbook anywhere; bad: trust only a data-directory rule."""
    p = Path(path)
    return (p.suffix.lower() in FORBIDDEN_SUFFIXES
            or any(part in PRIVATE_DIRS for part in p.parts)
            or (p.name.startswith('.env') and p.name != '.env.example')
            or '.private.' in p.name)


def scan(root: Path = ROOT, reference_workbook: Path | None = None) -> dict:
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = [name for name in files if name]
    if not files:
        raise ValueError('No tracked/staged files; stage the intended release before scanning')
    findings = []
    transcript_strings: set[str] = set()
    if reference_workbook is not None:
        import openpyxl
        book = openpyxl.load_workbook(reference_workbook, read_only=True, data_only=True)
        try:
            for sheet in book:
                rows = sheet.iter_rows(values_only=True)
                headers = next(rows, ())
                column = next((i for i, name in enumerate(headers) if name == 'Transcript'), None)
                if column is not None:
                    for row in rows:
                        if column < len(row) and row[column] is not None:
                            text = ' '.join(str(row[column]).casefold().split())
                            if len(text) >= 30:
                                transcript_strings.add(text)
        finally:
            book.close()
    for name in files:
        path = root / name
        if prohibited(name):
            findings.append({'path': name, 'kind': 'private_or_binary_path'})
        if path.is_symlink():
            findings.append({'path': name, 'kind': 'symlink_requires_review'})
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append({'path': name, 'kind': 'unexpected_binary'})
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append({'path': name, 'kind': 'credential_pattern'})
        if path.suffix == '.ipynb':
            notebook = json.loads(text)
            if any(c.get('outputs') or c.get('execution_count') is not None
                   for c in notebook.get('cells', []) if c.get('cell_type') == 'code'):
                findings.append({'path': name, 'kind': 'executed_notebook'})
        normalized = ' '.join(text.casefold().split())
        overlap = sum(t in normalized for t in transcript_strings)
        if overlap:
            findings.append({'path': name, 'kind': 'exact_long_transcript_match', 'count': overlap})
    # New repos may be staged but not yet committed. Do not inspect unrelated repositories.
    has_head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root,
                              capture_output=True).returncode == 0
    history_paths: set[str] = set()
    if has_head:
        objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=root, text=True)
        history_paths = {row.split(' ', 1)[1] for row in objects.splitlines() if ' ' in row}
        for name in sorted(history_paths):
            if prohibited(name):
                findings.append({'path': name, 'kind': 'private_path_in_git_history'})
    return {'passed': not findings, 'tracked_files': len(files),
            'history_paths_checked': len(history_paths),
            'reference_long_transcript_strings': len(transcript_strings),
            'findings': findings,
            'limits': 'Exact-match/path/pattern checks only; human privacy review remains required.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-workbook', type=Path)
    result = scan(reference_workbook=parser.parse_args().reference_workbook)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
