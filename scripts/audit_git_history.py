#!/usr/bin/env python3
"""Read-only privacy checks over every blob reachable from selected Git refs.

Good: ``uv run python scripts/audit_git_history.py --reference-workbook /private/coded.xlsx``.
Bad: scanning only HEAD and calling it a history audit, or uploading reference data.
Reference files are read locally; findings never include matched private text.
This is a bounded detector, not a guarantee that arbitrary sensitive data is absent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

from check_repository import SECRET_PATTERNS, prohibited

EXTRA_BINARY_SUFFIXES = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsm',
                         '.pkl', '.pickle', '.zip', '.7z', '.tar', '.gz'}
WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def tokens(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize('NFKC', text).casefold())


def private_phrases(workbooks: list[Path], codebooks: list[Path], words: int) -> set[tuple[str, ...]]:
    """Build local exact-token windows, without returning participant text in reports.

    Good: private source workbooks and original codebook examples.
    Bad: treating a zero-match result as proof about paraphrases or very short text.
    """
    import openpyxl
    import yaml

    phrases: set[tuple[str, ...]] = set()

    def include(value):
        if not isinstance(value, str):
            return
        parts = tokens(value)
        for i in range(len(parts) - words + 1):
            window = tuple(parts[i:i + words])
            if sum(map(len, window)) >= 40:
                phrases.add(window)

    for path in workbooks:
        book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        try:
            for sheet in book:
                rows = sheet.iter_rows(values_only=True)
                header = next(rows, ())
                columns = [i for i, value in enumerate(header)
                           if isinstance(value, str) and value.strip().casefold() in
                           ('transcript', 'utterance', 'text', 'turn text')]
                for row in rows:
                    for column in columns:
                        if column < len(row):
                            include(row[column])
        finally:
            book.close()
    for path in codebooks:
        obj = yaml.safe_load(path.read_text(encoding='utf-8'))

        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('positive_examples', 'negative_examples'):
                        for example in item or []:
                            include(example.get('text') if isinstance(example, dict) else example)
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(obj)
    return phrases


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def inspect_text(data: bytes, paths: list[str], oid: str,
                 phrases: set[tuple[str, ...]], words: int) -> list[dict]:
    """Return locations and detector names, never source text or secret values.

    Good: inspecting the actual historical blob, not its current file contents.
    Bad: putting the matching paragraph into a public audit report.
    """
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return []
    findings = []
    for index, pattern in enumerate(SECRET_PATTERNS):
        if pattern.search(text):
            findings.append({'kind': 'secret_pattern', 'pattern': index, 'blob': oid, 'paths': paths})
    if any(Path(path).suffix.casefold() == '.ipynb' for path in paths):
        try:
            notebook = json.loads(text)
            cells = notebook.get('cells', [])
            if any(c.get('outputs') or c.get('execution_count') is not None for c in cells):
                findings.append({'kind': 'executed_notebook', 'blob': oid, 'paths': paths})
        except (ValueError, AttributeError, TypeError):
            findings.append({'kind': 'invalid_notebook', 'blob': oid, 'paths': paths})
    if phrases:
        # JSON escapes should not hide notebook/output strings or JSON payloads.
        variants = [text]
        try:
            value = json.loads(text)

            def strings(obj):
                if isinstance(obj, str):
                    variants.append(obj)
                elif isinstance(obj, list):
                    for item in obj:
                        strings(item)
                elif isinstance(obj, dict):
                    for item in obj.values():
                        strings(item)

            strings(value)
        except (ValueError, RecursionError):
            pass
        for variant in variants:
            parts = tokens(variant)
            match = next((tuple(parts[i:i + words]) for i in range(len(parts) - words + 1)
                          if tuple(parts[i:i + words]) in phrases), None)
            if match:
                findings.append({'kind': 'private_reference_text', 'blob': oid, 'paths': paths,
                                 'match_sha256': hashlib.sha256(' '.join(match).encode()).hexdigest()})
                break
    return findings


def audit(root: Path, refs: list[str] | None = None,
          phrases: set[tuple[str, ...]] | None = None, words: int = 10) -> dict:
    """Scan reachable commit trees and blob contents, with Git/source files unmodified.

    Good: audit a fresh remote mirror to establish published reachability.
    Bad: assuming unreachable server objects, LFS payloads, releases or caches are covered.
    """
    selection = refs or ['--all']
    # Verify caller refs before passing them to revision traversal.
    for ref in refs or []:
        if ref.startswith('-'):
            raise ValueError('refs must be explicit Git ref names or object IDs')
        git(root, 'rev-parse', '--verify', ref + '^{commit}')
    commits = git(root, 'rev-list', *selection).decode().splitlines()
    if not commits:
        raise ValueError('no reachable commits to audit')
    paths_by_blob: dict[str, set[str]] = defaultdict(set)
    symlinks: set[tuple[str, str]] = set()
    for commit in commits:
        entries = git(root, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0')
        for entry in filter(None, entries):
            metadata, raw_path = entry.split(b'\t', 1)
            mode, kind, oid = metadata.decode().split()
            path = raw_path.decode('utf-8', errors='surrogateescape')
            if kind == 'blob':
                paths_by_blob[oid].add(path)
                if mode == '120000':
                    symlinks.add((oid, path))
            elif mode == '160000':
                # Submodule contents are not covered by this repository scan.
                paths_by_blob['submodule:' + oid].add(path)
    findings = []
    for oid, path in sorted(symlinks):
        findings.append({'kind': 'symlink_requires_review', 'blob': oid, 'paths': [path]})
    for oid, paths in sorted(paths_by_blob.items()):
        for path in sorted(paths):
            p = Path(path)
            if (prohibited(path)
                    or p.suffix.casefold() in EXTRA_BINARY_SUFFIXES
                    or '.private.' in p.name
                    or p.name in ('.env', '.env.local', 'credentials.json', 'token.json')):
                findings.append({'kind': 'protected_path', 'blob': oid, 'paths': [path]})
            if oid.startswith('submodule:'):
                findings.append({'kind': 'unscanned_submodule', 'blob': oid, 'paths': [path]})
    # Batch the exact historical contents, including commit messages.
    objects = [oid for oid in paths_by_blob if not oid.startswith('submodule:')] + commits
    if objects:
        process = subprocess.run(['git', '-C', str(root), 'cat-file', '--batch'],
                                 input=('\n'.join(objects) + '\n').encode(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data = process.stdout
        offset = 0
        for expected in objects:
            end = data.index(b'\n', offset)
            oid, kind, size = data[offset:end].decode().split()
            if oid != expected or not size.isdigit():
                raise ValueError('Git object stream identity mismatch')
            start = end + 1
            content = data[start:start + int(size)]
            offset = start + int(size) + 1
            paths = sorted(paths_by_blob.get(oid, {'<commit-message>'}))
            if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
                findings.append({'kind': 'unscanned_lfs_payload', 'blob': oid, 'paths': paths})
            elif content.startswith((b'PK\x03\x04', b'%PDF-', b'SQLite format 3\x00')):
                findings.append({'kind': 'embedded_protected_binary', 'blob': oid, 'paths': paths})
            else:
                try:
                    content.decode('utf-8')
                except UnicodeDecodeError:
                    findings.append({'kind': 'unreviewed_binary', 'blob': oid, 'paths': paths})
            findings.extend(inspect_text(content, paths, oid, phrases or set(), words))
    refs_snapshot = git(root, 'for-each-ref', '--format=%(refname) %(objectname)').decode().splitlines()
    return {'schema': 'git-privacy-audit/1', 'passed': not findings,
            'selection': selection, 'refs_snapshot': refs_snapshot,
            'commits_checked': len(commits),
            'unique_blobs_checked': sum(not oid.startswith('submodule:') for oid in paths_by_blob),
            'reference_phrase_windows': len(phrases or []), 'phrase_words': words,
            'findings': findings,
            'limits': ['Reachable Git objects only; no server caches, forks, releases or deleted refs.',
                       'Exact normalized token windows only; paraphrases and short excerpts may escape detection.',
                       'Secret patterns are bounded; participant identifiers need separate contextual review.']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ref', action='append')
    parser.add_argument('--reference-workbook', type=Path, action='append', default=[])
    parser.add_argument('--reference-codebook', type=Path, action='append', default=[])
    parser.add_argument('--phrase-words', type=int, default=10)
    args = parser.parse_args(argv)
    if args.phrase_words < 6:
        parser.error('--phrase-words must be at least 6')
    try:
        phrases = private_phrases(args.reference_workbook, args.reference_codebook, args.phrase_words)
        report = audit(args.repository, args.ref, phrases, args.phrase_words)
    except Exception:
        # Parser diagnostics can include private source lines; do not echo them.
        print(json.dumps({'passed': False, 'error': 'audit_failed; no clean-audit claim is valid'}))
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
