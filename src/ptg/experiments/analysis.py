"""Private analyses; public presentation has a deliberately small typed allowlist."""
from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

from .contract import ROOT, VERSION, canonical, keys, recipe_names, require
from .policy import codes
from .suite import (ledger_path, private_directory, read_private, unseal, validate_ledger,
                    verified, write_private, repetition_registry)


def agreement(left, right):
    require(set(left) == set(right), 'agreement populations differ')
    covered = [k for k in left if left[k] is not None and right[k] is not None]
    return {'eligible': len(left), 'covered': len(covered),
            'coverage': len(covered) / len(left) if left else None,
            'agreement_on_covered': sum(left[k] == right[k] for k in covered) / len(covered)
            if covered else None, 'equal_including_abstentions': sum(left[k] == right[k] for k in left)}


def vectors(values, labels):
    rows = {r for r, _ in values}
    return {r: tuple(values[(r, l)] for l in labels)
            if all(values[(r, l)] is not None for l in labels) else None for r in rows}


def comparison(left, right, labels):
    return {'per_label': {l: agreement({r: v for (r, x), v in left.items() if x == l},
                                      {r: v for (r, x), v in right.items() if x == l}) for l in labels},
            'complete_label_vector': agreement(vectors(left, labels), vectors(right, labels))}


def score(values, gold):
    require(set(values) == set(gold), 'scoring populations differ')
    counts = dict.fromkeys(('tp', 'tn', 'fp', 'fn', 'abstained_positive', 'abstained_negative'), 0)
    for k, v in values.items():
        g = gold[k]
        counts[('abstained_positive' if g else 'abstained_negative') if v is None else
               ('tp' if g else 'fp') if v else ('fn' if g else 'tn')] += 1
    n = len(values)
    return {**counts, 'eligible': n, 'coverage': sum(v is not None for v in values.values()) / n if n else None}


def report(out, repeats):
    plan = verified(out)
    require(repeats and len(set(repeats)) == len(repeats), 'distinct repetitions required')
    require(set(repeats) == set(repetition_registry(out, plan)['names']),
            'report must include every registered repetition; incomplete or quarantined runs cannot be omitted')
    labels = plan['panel']['bundle']['labels']
    all_values, summaries = {}, {}
    gold = {(j['row'], l): g for j in plan['jobs'] for l, g in zip(j['labels'], j['gold'], strict=True)}
    for repeat in repeats:
        ledger = validate_ledger(plan, unseal(read_private(ledger_path(out, repeat))), repeat)
        require(ledger['completion'] is not None, 'incomplete/unvalidated run cannot be reported')
        values = {arm: {} for arm in plan['arms']}
        for j in plan['jobs']:
            values[j['arm']].update({(j['row'], l): v for l, v in
                zip(j['labels'], ledger['decisions'][j['key']]['values'], strict=True)})
        all_values[repeat] = values
        summaries[repeat] = {'arms': {a: {'per_label': {l: score({k: v for k, v in vs.items() if k[1] == l},
                    {k: g for k, g in gold.items() if k[1] == l}) for l in labels},
                    'overall': score(vs, gold)} for a, vs in values.items()},
                    'attempt_outcomes': {outcome: sum(a['outcome'] == outcome for a in ledger['attempts'])
                                        for outcome in ('ok', 'invalid', 'transport', 'interrupted')},
                    'sessions': len(ledger['sessions'])}
    repeatability = {a: {x + '/' + y: comparison(all_values[x][a], all_values[y][a], labels)
                        for x, y in itertools.combinations(repeats, 2)} for a in plan['arms']}
    sensitivity = {rep: {a + '/' + b: comparison(v[a], v[b], labels)
                        for a, b in itertools.combinations(plan['arms'], 2)} for rep, v in all_values.items()}
    targets = [r for r in plan['panel']['bundle']['rows'] if r['target']]
    coder_count = len(targets[0]['coder_codes'])
    require(all(len(r['coder_codes']) == coder_count for r in targets), 'coder panel count differs')
    human = {}
    for i in range(coder_count):
        coder = {(r['id'], l): int(l in codes(r['coder_codes'][i], labels)) for r in targets for l in labels}
        human[str(i)] = {rep: {a: comparison(vs, coder, labels) for a, vs in arms.items()}
                         for rep, arms in all_values.items()}
    result = {'schema': VERSION, 'synthetic': plan['panel']['bundle']['synthetic'],
              'repetitions': summaries, 'repeatability': repeatability,
              'classifier_sensitivity': sensitivity, 'individual_coder_agreement': human}
    write_private(Path(out) / 'analysis.json', result)
    return {'validated_repetitions': len(repeats), 'arms': len(plan['arms']),
            'jobs_per_repetition': len(plan['jobs']), 'synthetic': result['synthetic']}


def public_metadata(plan):
    # Never emit classifier/input/source digests, IDs, host details, or rare-cell metrics.
    result = {'schema': VERSION, 'recipe_names': list(plan['arms']),
              'evidence_kind': 'synthetic' if plan['panel']['bundle']['synthetic'] else 'private-unpublished',
              'phase': 'prepared'}
    validate_public(result)
    return result


def validate_public(value):
    keys(value, 'schema recipe_names evidence_kind phase')
    require(value['schema'] == VERSION and isinstance(value['recipe_names'], list) and
            all(n in recipe_names() for n in value['recipe_names']) and
            value['evidence_kind'] in ('synthetic', 'private-unpublished') and
            value['phase'] == 'prepared', 'public allowlist violation')
    return value


PUBLIC_SOURCE_ALLOWLIST = {
    'scripts/classifier_suite_v2.py', 'config/experiments/suite-v2.json',
    'docs/classifier-suite-v2.md', 'tests/test_classifier_suite_v2.py',
    *{'src/ptg/experiments/' + n + '.py' for n in
      ('__init__', 'contract', 'policy', 'adapters', 'suite', 'analysis', 'cli')},
}


def export_sources(paths, out, root=ROOT):
    """Explicit file selection only. Git ignore alone is never a packaging boundary."""
    require(paths and len(set(paths)) == len(paths), 'explicit unique source list required')
    payload = {}
    for name in paths:
        require(name in PUBLIC_SOURCE_ALLOWLIST, 'source export allowlist rejection')
        path = root / name
        require(not any(p.is_symlink() for p in [path, *path.parents]), 'export symlink rejected')
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', name], cwd=root,
                                 capture_output=True).returncode == 0
        ignored = subprocess.run(['git', 'check-ignore', '--no-index', '-q', '--', name], cwd=root,
                                 capture_output=True).returncode == 0
        require(tracked and not ignored, 'untracked/ignored source export rejected')
        committed = subprocess.run(['git', 'show', 'HEAD:' + name], cwd=root, capture_output=True)
        require(committed.returncode == 0 and committed.stdout == path.read_bytes(),
                'source export requires committed unchanged bytes')
        payload[name] = committed.stdout.decode('utf-8')
    directory = private_directory(out, fresh=True, create=True)
    write_private(directory / 'public-source-bundle.json', payload)
    return {'exported_source_files': len(payload)}
