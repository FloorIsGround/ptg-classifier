"""Frozen preparation and durable attempts with explicit fresh-repeat namespaces."""
from __future__ import annotations

import copy
import fcntl
import json
import os
import math

import httpx
from pathlib import Path
import re
import uuid

from .adapters import RuntimeDrift, SYSTEM_TEXT, TransportFailure, admit, validate_reply
from .contract import (ROOT, VERSION, ContractError, canonical, digest, implementation_identity,
                       integer, keys, recipes, require, runtime_valid)
from .policy import codes, context_for, retrieved_for, validate_bundle


def private_directory(path, fresh=False, create=False):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in [path, *path.parents]), 'private symlink ancestor refused')
    path = path.resolve()
    base = (ROOT / 'artifacts').resolve()
    require(base in path.parents and path != base and not any(p.is_symlink() for p in
            [path, *path.parents]), 'private output must be nonsymlink worktree artifacts descendant')
    require(not fresh or not path.exists(), 'fresh namespace already exists')
    # Create parents individually with restrictive modes from the outset.
    require(create or path.exists(), 'private output does not exist')
    chain = []
    p = path
    while not p.exists():
        chain.append(p); p = p.parent
    for p in reversed(chain):
        p.mkdir(mode=0o700)
    require(path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o777 == 0o700, 'private directory permissions differ')
    return path


def write_private(path, value):
    require(not path.is_symlink(), 'private symlink refused')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_private(path):
    require(not path.is_symlink() and path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o777 == 0o600,
            'private file permissions/symlink differ')
    return json.loads(path.read_text())


def seal(value):
    return {**value, 'seal': digest(value)}


def unseal(value):
    require(isinstance(value, dict) and 'seal' in value, 'missing artifact seal')
    value = value.copy(); expected = value.pop('seal')
    require(digest(value) == expected, 'artifact tampering/drift')
    return value


def prepare(bundle, names, runtimes, out=None, factorial=False):
    targets, sources = validate_bundle(bundle)
    registry = recipes()
    require(names and len(set(names)) == len(names) and all(n in registry for n in names),
            'unique registered recipes required')
    require(set(runtimes) == set(names), 'runtime bindings differ from recipes')
    selected = [registry[n] for n in names]
    # Sequential panels by default. Factorial must be explicitly declared before preparation.
    if not factorial:
        varying_context = len({digest(r['context']) for r in selected}) > 1
        varying_retrieval = len({digest(r['retrieval']) for r in selected}) > 1
        require(not (varying_context and varying_retrieval), 'context/retrieval confounded; declare factorial')
        require(len({digest({k: v for k, v in r.items() if k not in
                            ('name', 'context', 'retrieval')}) for r in selected}) == 1,
                'model/packing comparison requires explicit factorial declaration')
    impl = implementation_identity()
    panel = {'bundle': bundle, 'labels': bundle['labels'], 'factorial': factorial,
             'serialization': 'canonical-json-utf8/2', 'normalization': 'strip-only/2',
             'blank_policy': 'observed-negative', 'scoring': 'abstention-preserving/2'}
    panel_fp = digest(panel)
    arms, jobs = {}, []
    for r in selected:
        runtime = runtime_valid(runtimes[r['name']])
        require(runtime['backend'] == r['backend'] and runtime['model'] == r['model'],
                'model/runtime binding differs')
        classifier = {'schema': VERSION, 'recipe': r, 'runtime': runtime, 'implementation': impl,
                      'labels': bundle['labels'], 'codebook': bundle['codebook'],
                      'retrieval_sources': bundle['sources'], 'retrieval_identity': bundle['retrieval_identity'],
                      'serialization': panel['serialization'], 'normalization': panel['normalization'],
                      'validation': 'strict-completed-binary-or-native-probability/2',
                      'blank_policy': panel['blank_policy'], 'scoring': panel['scoring']}
        arms[r['name']] = {'classifier': classifier, 'fingerprint': digest(classifier)}
        for row in sorted(targets, key=lambda r: r['id']):
            context = context_for(bundle['rows'], row, r['context'])
            questions = []
            for e in bundle['codebook']:
                questions.append({'label': e['label'], 'definition': e['definition'],
                                  'positive': e['positive'], 'negative': e['negative'],
                                  'retrieved': retrieved_for(bundle, sources, row, e['label'], r['retrieval']),
                                  'question': 'Does ONLY the target exhibit this label?'})
            groups = [questions] if r['packing'] == 'shared_state' else [[q] for q in questions]
            for group in groups:
                state = {'context': context, 'target': row['text'].strip(),
                         'attribution': 'Only the target is evidence; prior turns clarify meaning.'}
                # Native serializer is state + ordered questions; zero generated output tokens.
                wire = canonical({'state': state, 'questions': group}).decode()
                if r['backend'] != 'd1_native':
                    wire += '\nAnswer only yes or no.'
                upper = len(wire.encode()) + len(SYSTEM_TEXT.encode()) + runtime['template_overhead_tokens'] + r['decoding']['num_predict']
                require(upper < r['decoding']['num_ctx'], 'prompt token upper-bound overflow before inference')
                labels = [q['label'] for q in group]
                job = {'arm': r['name'], 'row': row['id'], 'labels': labels,
                       'gold': [int(l in codes(row['gold'], bundle['labels'])) for l in labels],
                       'state': state, 'questions': group, 'wire': wire, 'token_upper_bound': upper}
                job['key'] = digest([job['arm'], job['row'], labels])
                job['input_fingerprint'] = digest(job)
                jobs.append(job)
    # Every arm must cover exactly the frozen rows x labels with identical gold.
    expected = {(r['id'], l): int(l in codes(r['gold'], bundle['labels']))
                for r in targets for l in bundle['labels']}
    for n in names:
        actual = {(j['row'], l): g for j in jobs if j['arm'] == n
                  for l, g in zip(j['labels'], j['gold'], strict=True)}
        require(actual == expected, 'per-arm population/gold drift')
    jobs.sort(key=lambda job: job['key'])
    plan = {'schema': VERSION, 'panel': panel, 'panel_fingerprint': panel_fp,
            'arms': arms, 'jobs': jobs, 'implementation': impl}
    if out is not None:
        directory = private_directory(out, fresh=True, create=True)
        try:
            write_private(directory / 'prepared.json', seal(plan))
        except BaseException:
            directory.rmdir()
            raise
    return plan


def verified(out):
    directory = private_directory(out)
    require(not (directory / 'quarantine.json').exists(), 'prepared panel quarantined; new preparation required')
    plan = unseal(read_private(directory / 'prepared.json'))
    require(plan['implementation'] == implementation_identity(), 'source/dependency drift; new preparation required')
    rebuilt = prepare(plan['panel']['bundle'], list(plan['arms']),
                      {n: a['classifier']['runtime'] for n, a in plan['arms'].items()},
                      factorial=plan['panel']['factorial'])
    require(plan == rebuilt, 'prepared classifier/panel/input drift')
    return plan


def ledger_path(out, repeat):
    require(isinstance(repeat, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,40}', repeat),
            'invalid repetition namespace')
    return Path(out) / ('repeat-' + repeat + '.json')


def validate_ledger(plan, ledger, repeat):
    keys(ledger, 'schema namespace run_id plan_fingerprint attempts decisions quarantine completion sessions')
    require(ledger['schema'] == VERSION and ledger['namespace'] == repeat and
            ledger['plan_fingerprint'] == digest(plan), 'replay/run namespace or identity drift')
    require(not ledger['quarantine'], 'runtime drift quarantined; fresh preparation required')
    jobs = {j['key']: j for j in plan['jobs']}
    require(isinstance(ledger['decisions'], dict) and set(ledger['decisions']) <= set(jobs),
            'extra decision keys')
    grouped = {}
    for a in ledger['attempts']:
        keys(a, 'id key input_fingerprint ordinal outcome response parsed')
        require(a['key'] in jobs and a['input_fingerprint'] == jobs[a['key']]['input_fingerprint'],
                'attempt input/key drift')
        group = grouped.setdefault(a['key'], [])
        require(a['ordinal'] == len(group) + 1 and a['outcome'] in
                ('started', 'ok', 'invalid', 'transport', 'interrupted', 'drift'), 'attempt order/outcome drift')
        group.append(a)
        if a['outcome'] in ('ok', 'invalid'):
            r = plan['arms'][jobs[a['key']]['arm']]['classifier']['recipe']
            response = a['response'].get('validated_response', a['response'])
            if 'wire_response' in a['response']:
                raw = a['response']['wire_response']
                projected = {'model': raw.get('model'), 'done': raw.get('done'),
                    'done_reason': raw.get('done_reason'), 'message': {
                    'content': raw['message'].get('content'), 'thinking': raw['message'].get('thinking', '')},
                    'prompt_eval_count': raw.get('prompt_eval_count'), 'eval_count': raw.get('eval_count')}
                require(projected == response, 'wire/validated response differs')
            try:
                parsed = validate_reply(response, jobs[a['key']], r)
            except (ContractError, ValueError, KeyError, TypeError):
                require(a['outcome'] == 'invalid', 'invalid accepted response')
            else:
                require((a['outcome'] == 'ok' and None not in parsed and a['parsed'] == parsed) or
                        (a['outcome'] == 'invalid' and None in parsed), 'response outcome/parsing drift')
    for key, decision in ledger['decisions'].items():
        keys(decision, 'attempt_id values')
        attempts = grouped.get(key, [])
        require(attempts and decision['attempt_id'] == attempts[-1]['id'], 'decision provenance differs')
        a = attempts[-1]
        require(a['outcome'] in ('ok', 'invalid'), 'decision cannot represent infrastructure failure')
        require(decision['values'] == (a['parsed'] if a['outcome'] == 'ok' else
                [None] * len(jobs[key]['labels'])), 'decision/abstention drift')
    if ledger['completion'] is not None:
        payload = {k: v for k, v in ledger.items() if k != 'completion'}
        require(len(ledger['decisions']) == len(jobs) and ledger['completion'] == digest(payload),
                'completion validation differs')
    return ledger


def audit_response(value):
    """Preserve invalid IEEE numeric values without writing nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'invalid_ieee_float': repr(value)}
    if isinstance(value, list):
        return [audit_response(v) for v in value]
    if isinstance(value, dict):
        return {k: audit_response(v) for k, v in value.items()}
    return value


def repetition_registry(out, plan):
    path = Path(out) / 'repetitions.json'
    if not path.exists():
        return {'plan_fingerprint': digest(plan), 'names': []}
    registry = unseal(read_private(path))
    keys(registry, 'plan_fingerprint names')
    require(registry['plan_fingerprint'] == digest(plan) and
            len(set(registry['names'])) == len(registry['names']), 'repetition registry drift')
    return registry


def run(out, repeat, adapter, resume=False):
    plan = verified(out)
    synthetic = plan['panel']['bundle']['synthetic']
    require(adapter.synthetic == synthetic, 'synthetic/real backend boundary differs')
    require(synthetic or len(plan['jobs']) <= 512, 'native first-phase panel limited to 512 jobs; scalable journal required')
    for arm in plan['arms'].values():
        admit(arm['classifier']['recipe'], synthetic)
    # One runtime per execution panel prevents accidental endpoint/model substitutions.
    require(len({digest(a['classifier']['runtime']) for a in plan['arms'].values()}) == 1,
            'run model panels separately')
    expected = next(iter(plan['arms'].values()))['classifier']['runtime']
    path = ledger_path(out, repeat)
    lockpath = Path(out) / 'run.lock'
    require(not lockpath.is_symlink(), 'lock symlink refused')
    fd = os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(fd).st_uid == os.getuid() and os.fstat(fd).st_mode & 0o777 == 0o600,
                'private lock permissions differ')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(path.exists() if resume else not path.exists(), 'resume requires existing run; fresh repeat requires new namespace')
        ledger = validate_ledger(plan, unseal(read_private(path)), repeat) if resume else {
            'schema': VERSION, 'namespace': repeat, 'run_id': uuid.uuid4().hex,
            'plan_fingerprint': digest(plan), 'attempts': [], 'decisions': {},
            'quarantine': False, 'completion': None, 'sessions': []}
        registry = repetition_registry(out, plan)
        require((repeat in registry['names']) == resume, 'repetition registry/namespace differs')
        if not resume:
            registry['names'].append(repeat)
            write_private(Path(out) / 'repetitions.json', seal(registry))
        start_count = len(ledger['attempts'])
        ledger['completion'] = None
        ledger['sessions'].append({'id': uuid.uuid4().hex, 'kind': 'resume' if resume else 'fresh'})
        def save():
            write_private(path, seal(ledger))
        def check_runtime():
            try:
                require(adapter.identity() == expected, 'runtime identity differs')
            except httpx.HTTPError as exc:
                raise TransportFailure('identity request transport failure') from exc
            except Exception as exc:
                ledger['quarantine'] = True
                write_private(Path(out) / 'quarantine.json', {'reason': 'runtime_identity_drift'})
                save()
                raise RuntimeDrift('runtime identity failed; quarantined') from exc
        save(); check_runtime()
        for a in ledger['attempts']:
            if a['outcome'] == 'started':
                a['outcome'] = 'interrupted'
        save()
        for job in sorted(plan['jobs'], key=lambda j: digest([42, j['key']])):
            key = job['key']
            if key in ledger['decisions']:
                continue
            recipe = plan['arms'][job['arm']]['classifier']['recipe']
            prior = [a for a in ledger['attempts'] if a['key'] == key]
            # Recover a persisted final response without making another inference call.
            if not prior or prior[-1]['outcome'] in ('transport', 'interrupted'):
                session_attempts = 0
                while session_attempts < recipe['retry']['max_attempts']:
                    session_attempts += 1
                    attempt = {'id': uuid.uuid4().hex, 'key': key,
                               'input_fingerprint': job['input_fingerprint'], 'ordinal': len(prior) + 1,
                               'outcome': 'started', 'response': None, 'parsed': None}
                    ledger['attempts'].append(attempt); prior.append(attempt); save()
                    try:
                        check_runtime()
                        response = adapter.infer(copy.deepcopy(job), copy.deepcopy(recipe))
                        attempt['response'] = audit_response(response)
                        check_runtime()
                        attempt['parsed'] = validate_reply(response.get('validated_response', response), job, recipe)
                        attempt['outcome'] = 'ok' if None not in attempt['parsed'] else 'invalid'
                    except RuntimeDrift:
                        attempt['outcome'] = 'drift'; ledger['quarantine'] = True
                        write_private(Path(out) / 'quarantine.json', {'reason': 'response_identity_drift'})
                        save(); raise
                    except TransportFailure as exc:
                        attempt['outcome'] = 'transport'; attempt['response'] = audit_response(exc.audit)
                    except (ContractError, ValueError, KeyError, TypeError) as exc:
                        attempt['outcome'] = 'invalid'
                        if attempt['response'] is None:
                            attempt['response'] = {'error_type': type(exc).__name__}
                    save()
                    if attempt['outcome'] != 'transport':
                        break
            require(prior, 'missing attempt')
            last = prior[-1]
            if last['outcome'] in ('transport', 'interrupted', 'started'):
                save()
                raise TransportFailure('request exhausted transport budget; incomplete run can resume')
            ledger['decisions'][key] = {'attempt_id': last['id'], 'values': last['parsed']
                if last['outcome'] == 'ok' else [None] * len(job['labels'])}
            save()
        check_runtime()
        ledger['completion'] = digest({k: v for k, v in ledger.items() if k != 'completion'})
        save(); validate_ledger(plan, ledger, repeat)
        return {'completed_jobs': len(ledger['decisions']), 'attempts': len(ledger['attempts']),
                'new_inference_attempts': len(ledger['attempts']) - start_count, 'synthetic': synthetic}
    finally:
        os.close(fd)
