"""Fictional-only v2 contract, execution and privacy regression tests."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

from ptg.experiments.adapters import (AdmissionBlocked, ContractError, OllamaAdapter,
    RuntimeDrift, SyntheticAdapter, TransportFailure, admit, loopback, probability_response)
from ptg.experiments.analysis import (comparison, export_sources, public_metadata, report,
                                     validate_public)
from ptg.experiments.cli import main, synthetic_bundle, synthetic_runtime
from ptg.experiments.contract import ROOT, digest, recipe_valid, recipes
from ptg.experiments.policy import codes, context_for
from ptg.experiments.suite import (prepare, read_private, run, seal, unseal, verified,
                                  write_private)


@pytest.fixture
def out():
    p = ROOT / 'artifacts' / ('synthetic-test-' + uuid.uuid4().hex)
    yield p
    if p.exists():
        shutil.rmtree(p)


def prepared(out, names=('rag_baseline', 'no_retrieval'), factorial=False):
    registry = recipes()
    bundle = synthetic_bundle()
    bindings = {n: synthetic_runtime(registry[n]) for n in names}
    plan = prepare(bundle, list(names), bindings, out, factorial)
    return plan, SyntheticAdapter(bindings[names[0]])


def test_registry_and_unknown_contract_keys():
    rs = recipes()
    assert len(rs) == 9
    r = copy.deepcopy(rs['rag_baseline']); r['unexpected'] = True
    with pytest.raises(ContractError):
        recipe_valid(r)


def test_static_context_and_population_matched(out):
    plan, _ = prepared(out)
    arms = {a: {(j['row'], tuple(j['labels'])): j for j in plan['jobs'] if j['arm'] == a}
            for a in plan['arms']}
    assert arms['rag_baseline'].keys() == arms['no_retrieval'].keys()
    for k, j in arms['rag_baseline'].items():
        control = arms['no_retrieval'][k]
        assert j['gold'] == control['gold']
        assert j['state'] == control['state']
        q, c = j['questions'][0].copy(), control['questions'][0].copy()
        assert q.pop('retrieved') and not c.pop('retrieved')
        assert q == c
    assert out.stat().st_mode & 0o777 == 0o700
    assert (out / 'prepared.json').stat().st_mode & 0o777 == 0o600


def test_blanks_are_no_codes_not_missing():
    assert codes('', ['x']) == []
    with pytest.raises(ContractError):
        codes(None, ['x'])
    with pytest.raises(ContractError):
        codes(['unknown'], ['x'])


@pytest.mark.parametrize('change', ['heldout', 'duplicate', 'order', 'unknown_gold', 'missing_coder',
                                   'source_heldout', 'source_development', 'source_overlap', 'static_overlap',
                                   'ranking', 'label_order'])
def test_bad_inputs_fail_before_output(out, change):
    b = synthetic_bundle()
    if change == 'heldout': b['rows'][0]['semester'] = 'FA26'
    if change == 'duplicate': b['rows'][1]['id'] = b['rows'][0]['id']
    if change == 'order': b['rows'][1]['position'] = 0
    if change == 'unknown_gold': b['rows'][1]['gold'] = ['unknown']
    if change == 'missing_coder': del b['rows'][1]['coder_codes']
    if change == 'source_heldout': b['sources'][0]['semester'] = 'FA26'
    if change == 'source_development': b['sources'][0]['semester'] = 'S26'
    if change == 'source_overlap': b['sources'][0]['text'] = b['rows'][1]['text']
    if change == 'static_overlap': b['codebook'][0]['positive'] = [b['rows'][1]['text']]
    if change == 'ranking': b['rankings'] = {}
    if change == 'label_order': b['labels'].reverse()
    r = recipes()['rag_baseline']
    with pytest.raises((ContractError, KeyError)):
        prepare(b, ['rag_baseline'], {'rag_baseline': synthetic_runtime(r)}, out)
    assert not out.exists()


def test_context_is_causal_meeting_bounded_and_decay_exact():
    b = synthetic_bundle()
    row = b['rows'][2]
    policy = recipes()['context_prev6']['context']
    context = context_for(b['rows'], row, policy)
    assert context == [b['rows'][0]['text'], b['rows'][1]['text']]
    assert row['text'] not in context
    assert b['rows'][3]['text'] not in context
    decay = {**policy, 'policy': 'recency_decay_budget', 'budget_bytes': 100, 'predecessors': 12}
    assert context_for(b['rows'], row, decay) == [b['rows'][1]['text']]
    assert context_for(b['rows'], row, decay) == context_for(list(reversed(b['rows'])), row, decay)


def test_overflow_atomic_prepare(out):
    b = synthetic_bundle(); b['rows'][-1]['text'] = 'Fictional ' * 3000
    r = recipes()['rag_baseline']
    with pytest.raises(ContractError, match='overflow'):
        prepare(b, ['rag_baseline'], {'rag_baseline': synthetic_runtime(r)}, out)
    assert not out.exists()


def test_sequential_context_panel_and_confound_gate(out):
    prepared(out, ('context_prev0', 'context_prev2', 'context_prev6', 'context_recency_decay_budget'))
    with pytest.raises(ContractError, match='confounded'):
        prepared(out.parent / (out.name + '-rejected'), ('rag_baseline', 'context_prev6'))


def test_fingerprints_cover_all_classifier_facets(out):
    plan, _ = prepared(out)
    classifier = plan['arms']['rag_baseline']['classifier']
    for key in classifier:
        changed = copy.deepcopy(classifier); changed[key] = {'fictional_changed': True}
        assert digest(changed) != digest(classifier)
    assert plan['arms']['rag_baseline']['fingerprint'] != plan['arms']['no_retrieval']['fingerprint']


def test_fresh_repeat_resume_and_report(out):
    plan, fake = prepared(out)
    assert run(out, 'one', fake)['new_inference_attempts'] == 168
    assert run(out, 'one', fake, resume=True)['new_inference_attempts'] == 0
    assert run(out, 'two', fake)['new_inference_attempts'] == 168
    with pytest.raises(ContractError): run(out, 'one', fake)
    with pytest.raises(ContractError): run(out, 'absent', fake, resume=True)
    assert report(out, ['one', 'two']) == {'validated_repetitions': 2, 'arms': 2,
                                         'jobs_per_repetition': 168, 'synthetic': True}
    analysis = read_private(out / 'analysis.json')
    repeat = analysis['repeatability']['rag_baseline']['one/two']
    assert repeat['complete_label_vector']['covered'] == 4
    assert repeat['complete_label_vector']['agreement_on_covered'] == 1
    assert len(repeat['per_label']) == 21
    assert 'individual_coder_agreement' in analysis and 'classifier_sensitivity' in analysis


def test_transport_logged_retried_and_invalid_not_retried(out):
    _, fake = prepared(out, ('no_retrieval',))
    real_infer = fake.infer
    count = 0
    def infer(j, r):
        nonlocal count
        count += 1
        if count == 1: raise TransportFailure('fictional')
        response = real_infer(j, r)
        if count == 2: response['message']['content'] = 'fictional invalid'
        return response
    fake.infer = infer
    result = run(out, 'one', fake)
    assert result['attempts'] == 85
    ledger = unseal(read_private(out / 'repeat-one.json'))
    assert sum(a['outcome'] == 'transport' for a in ledger['attempts']) == 1
    assert sum(a['outcome'] == 'invalid' for a in ledger['attempts']) == 1
    assert sum(None in d['values'] for d in ledger['decisions'].values()) == 1


def test_interrupted_write_ahead_resume(out):
    _, fake = prepared(out, ('no_retrieval',))
    infer = fake.infer
    fake.infer = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt): run(out, 'one', fake)
    ledger = unseal(read_private(out / 'repeat-one.json'))
    assert ledger['attempts'][0]['outcome'] == 'started'
    with pytest.raises(ContractError): report(out, ['one'])
    fake.infer = infer
    assert run(out, 'one', fake, resume=True)['completed_jobs'] == 84
    ledger = unseal(read_private(out / 'repeat-one.json'))
    assert ledger['attempts'][0]['outcome'] == 'interrupted'


def test_prepared_tampering_and_resealed_population_drift(out):
    prepared(out)
    path = out / 'prepared.json'
    value = read_private(path); value['jobs'].pop()
    write_private(path, value)
    with pytest.raises(ContractError, match='tampering'): verified(out)
    value.pop('seal'); write_private(path, seal(value))
    with pytest.raises(ContractError, match='drift'): verified(out)


def test_response_tamper_and_namespace_copy_rejected(out):
    _, fake = prepared(out)
    run(out, 'one', fake)
    ledger = unseal(read_private(out / 'repeat-one.json'))
    write_private(out / 'repeat-two.json', seal(ledger))
    with pytest.raises(ContractError, match='namespace'): run(out, 'two', fake, resume=True)
    ledger['attempts'][0]['parsed'] = [9]
    write_private(out / 'repeat-one.json', seal(ledger))
    with pytest.raises(ContractError): report(out, ['one'])


def test_runtime_drift_quarantines_at_end(out):
    _, fake = prepared(out, ('no_retrieval',))
    identity = fake.identity
    count = 0
    def changed():
        nonlocal count
        count += 1
        result = identity()
        if count > 168: result['engine_version'] = 'fictional-drift'
        return result
    fake.identity = changed
    with pytest.raises(RuntimeDrift): run(out, 'one', fake)
    assert unseal(read_private(out / 'repeat-one.json'))['quarantine']
    with pytest.raises(ContractError): report(out, ['one'])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True, '0.8'])
def test_invalid_native_probability(value):
    with pytest.raises(ContractError): probability_response({'labels': ['l'], 'noul': [value]}, ['l'])


def test_native_probability_order_length_and_threshold():
    assert probability_response({'labels': ['a', 'b', 'c'], 'noul': [0.499, 0.5, 1]}, ['a', 'b', 'c']) == [0, 1, 1]
    for response in [{'labels': ['b', 'a'], 'noul': [0.1, 0.9]},
                     {'labels': ['a', 'b'], 'noul': [0.9]},
                     {'labels': ['a', 'b'], 'noul': [0.1, 0.9], 'text': 'yes'}]:
        with pytest.raises(ContractError): probability_response(response, ['a', 'b'])


def test_d1_packing_is_distinct_and_native_gated(out):
    rs = recipes()
    single = rs['d1_single_label']; shared = rs['d1_shared_state']
    assert single['revision'] == '051bcc464b01b9f92942b364d9586b0ef5912432'
    assert digest(single) != digest(shared)
    for r in (single, shared):
        with pytest.raises(AdmissionBlocked): admit(r, False)
    plan, fake = prepared(out, ('d1_shared_state',))
    assert len(plan['jobs']) == 4
    assert all(len(j['questions']) == 21 for j in plan['jobs'])
    assert run(out, 'one', fake)['completed_jobs'] == 4
    assert report(out, ['one'])['jobs_per_repetition'] == 4
    with pytest.raises(AdmissionBlocked): admit(rs['granite42_confirmation'], True)


@pytest.mark.parametrize('url', ['http://example.invalid:11434', 'http://192.0.2.1:11434',
    'http://127.0.0.1:11434/path', 'http://u:p@127.0.0.1:11434', 'https://127.0.0.1:11434'])
def test_remote_origins_rejected(url):
    with pytest.raises(ValueError): loopback(url)


def test_ollama_native_wire_synthetic_transport(out):
    _, fake = prepared(out, ('no_retrieval',))
    runtime = fake.identity()
    adapter = OllamaAdapter('http://127.0.0.1:11434', runtime)
    def handler(request):
        import httpx
        if request.url.path == '/api/version': value = {'version': runtime['engine_version']}
        elif request.url.path == '/api/tags': value = {'models': [{'name': runtime['model'], 'digest': runtime['artifact_digest']}]}
        elif request.url.path == '/api/show': value = {'details': {'quantization_level': runtime['quantization']}}
        else:
            payload = json.loads(request.content)
            assert payload['think'] is False and payload['stream'] is False
            value = {'model': runtime['model'], 'done': True, 'done_reason': 'stop',
                     'message': {'content': 'yes'}, 'prompt_eval_count': 5, 'eval_count': 1}
        return httpx.Response(200, json=value)
    import httpx
    adapter.client.close(); adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    assert adapter.identity()['artifact_digest'] == runtime['artifact_digest']
    job = verified(out)['jobs'][0]
    assert adapter.infer(job, recipes()['no_retrieval'])['validated_response']['message']['content'] == 'yes'
    adapter.close()


def test_public_allowlist_never_exposes_private_metadata(out):
    plan, _ = prepared(out)
    metadata = public_metadata(plan)
    assert set(metadata) == {'schema', 'recipe_names', 'evidence_kind', 'phase'}
    assert 'fingerprint' not in json.dumps(metadata)
    for field in ('prompt', 'input_fingerprint', 'row_id', 'rare_cell', 'host'):
        with pytest.raises(ContractError): validate_public({**metadata, field: 'fictional-private'})
    with pytest.raises(ContractError): export_sources(['artifacts/private.json'], out / 'export')


def test_export_rejects_untracked_and_ignored_allowed_name(tmp_path, out):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    p = tmp_path / 'scripts/classifier_suite_v2.py'; p.parent.mkdir(); p.write_text('fictional')
    with pytest.raises(ContractError, match='untracked'): export_sources(['scripts/classifier_suite_v2.py'], out, tmp_path)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('scripts/\n')
    with pytest.raises(ContractError, match='ignored'): export_sources(['scripts/classifier_suite_v2.py'], out, tmp_path)


def test_coverage_excludes_abstained_vectors():
    ls = ['a', 'b']
    left = {('r', 'a'): 1, ('r', 'b'): None}
    right = {('r', 'a'): 1, ('r', 'b'): 0}
    c = comparison(left, right, ls)
    assert c['per_label']['a']['coverage'] == 1
    assert c['complete_label_vector']['coverage'] == 0
    assert c['complete_label_vector']['agreement_on_covered'] is None


def test_cli_synthetic_and_error_redaction(out, capsys):
    assert main(['prepare', '--synthetic', '--recipes', 'rag_baseline', 'no_retrieval', '--out', str(out)]) == 0
    assert main(['run', '--synthetic', '--out', str(out), '--repeat', 'first']) == 0
    assert main(['report', '--out', str(out), '--repeats', 'first']) == 0
    assert main(['run', '--synthetic', '--out', str(out), '--repeat', '/fictional-private']) == 2
    output = capsys.readouterr().out
    assert str(out) not in output and '/fictional-private' not in output


def test_exhausted_transport_is_incomplete_and_resumable(out):
    _, fake = prepared(out, ('no_retrieval',))
    infer = fake.infer
    fake.infer = lambda *args: (_ for _ in ()).throw(TransportFailure('fictional'))
    with pytest.raises(TransportFailure): run(out, 'one', fake)
    ledger = unseal(read_private(out / 'repeat-one.json'))
    assert len(ledger['attempts']) == 2 and not ledger['decisions'] and ledger['completion'] is None
    with pytest.raises(ContractError): report(out, ['one'])
    fake.infer = infer
    assert run(out, 'one', fake, resume=True)['completed_jobs'] == 84
    assert report(out, ['one'])['validated_repetitions'] == 1


def test_reports_cannot_omit_registered_repeats(out):
    _, fake = prepared(out, ('no_retrieval',))
    run(out, 'one', fake); run(out, 'two', fake)
    with pytest.raises(ContractError, match='every registered'): report(out, ['one'])


def test_quarantine_blocks_entire_prepared_panel(out):
    _, fake = prepared(out, ('no_retrieval',))
    fake.identity = lambda: {'fictional': 'mismatch'}
    with pytest.raises(RuntimeDrift): run(out, 'one', fake)
    assert (out / 'quarantine.json').exists()
    with pytest.raises(ContractError, match='quarantined'): run(out, 'two', fake)


def test_identity_transport_blip_stops_without_quarantine(out):
    import httpx
    _, fake = prepared(out, ('no_retrieval',))
    identity = fake.identity
    fake.identity = lambda: (_ for _ in ()).throw(httpx.ConnectError('fictional'))
    with pytest.raises(TransportFailure): run(out, 'one', fake)
    assert not (out / 'quarantine.json').exists()
    fake.identity = identity
    assert run(out, 'one', fake, resume=True)['completed_jobs'] == 84


def test_nonfinite_native_response_is_logged_abstention(out):
    _, fake = prepared(out, ('d1_shared_state',))
    fake.infer = lambda j, r: {'labels': j['labels'], 'noul': [float('nan')] * 21}
    assert run(out, 'one', fake)['completed_jobs'] == 4
    ledger = unseal(read_private(out / 'repeat-one.json'))
    assert all(a['outcome'] == 'invalid' for a in ledger['attempts'])
    assert all(v is None for d in ledger['decisions'].values() for v in d['values'])
    assert ledger['attempts'][0]['response']['noul'][0] == {'invalid_ieee_float': 'nan'}
    assert report(out, ['one'])['validated_repetitions'] == 1


def test_export_rejects_uncommitted_sensitive_edit(tmp_path, out):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    p = tmp_path / 'scripts/classifier_suite_v2.py'; p.parent.mkdir(); p.write_text('fictional source')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Fictional', '-c', 'user.email=fictional@example.invalid',
                    'commit', '-qm', 'fictional fixture'], cwd=tmp_path, check=True)
    p.write_text('fictional private sentinel inserted after commit')
    with pytest.raises(ContractError, match='committed'): export_sources(['scripts/classifier_suite_v2.py'], out, tmp_path)
    assert not out.exists()


def test_mistyped_read_output_does_not_create_directory(out):
    with pytest.raises(ContractError): verified(out)
    assert not out.exists()


def test_response_validation_failures_retain_abstentions(out):
    from ptg.experiments.adapters import validate_reply
    plan, fake = prepared(out, ('no_retrieval',))
    j = plan['jobs'][0]; r = recipes()['no_retrieval']
    response = fake.infer(j, r)
    for field, value in [('done', False), ('done_reason', 'length'),
                         ('prompt_eval_count', 8192), ('eval_count', 17)]:
        changed = copy.deepcopy(response); changed[field] = value
        with pytest.raises(ContractError): validate_reply(changed, j, r)
    changed = copy.deepcopy(response); changed['message']['thinking'] = 'fictional'
    with pytest.raises(ContractError): validate_reply(changed, j, r)
    changed = copy.deepcopy(response); changed['message']['content'] = 'yes.'
    assert validate_reply(changed, j, r) == [None]


def test_output_path_traversal_and_symlink_rejected(out):
    from ptg.experiments.suite import private_directory
    escape = ROOT / 'artifacts' / '..' / ('fictional-escape-' + uuid.uuid4().hex)
    with pytest.raises(ContractError): private_directory(escape, fresh=True, create=True)
    assert not escape.exists()
    symlink = ROOT / 'artifacts' / ('synthetic-link-' + uuid.uuid4().hex)
    symlink.symlink_to(ROOT, target_is_directory=True)
    try:
        with pytest.raises(ContractError): private_directory(symlink / 'fictional-child', fresh=True, create=True)
    finally:
        symlink.unlink()


def test_native_first_phase_panel_bound(out):
    b = synthetic_bundle(); b['synthetic'] = False
    # 13 target rows x21 labels x2 arms =546 jobs, above the admitted bound.
    template = b['rows'][1]
    b['rows'] = [{**copy.deepcopy(template), 'id': 'fictional-expanded-' + str(n),
                 'position': n, 'text': 'Fictional expanded target turn ' + str(n)} for n in range(13)]
    b['rankings'] = {r['id']: ['fictional-source'] for r in b['rows']}
    names = ['rag_baseline', 'no_retrieval']
    rt = {n: synthetic_runtime(recipes()[n]) for n in names}
    prepare(b, names, rt, out)
    fake = SyntheticAdapter(rt[names[0]]); fake.synthetic = False
    fake.identity = lambda: pytest.fail('must not touch runtime above admitted panel limit')
    with pytest.raises(ContractError, match='512'): run(out, 'one', fake)


def test_label_permutation_is_a_different_classifier():
    b = synthetic_bundle(); names = ['no_retrieval']
    rt = {'no_retrieval': synthetic_runtime(recipes()['no_retrieval'])}
    original = prepare(b, names, rt)
    b['labels'].reverse(); b['codebook'].reverse()
    permuted = prepare(b, names, rt)
    assert original['arms']['no_retrieval']['fingerprint'] != permuted['arms']['no_retrieval']['fingerprint']


def test_native_http_failure_keeps_private_body(out):
    import httpx
    _, fake = prepared(out, ('no_retrieval',))
    adapter = OllamaAdapter('http://127.0.0.1:11434', fake.identity())
    adapter.client.close()
    adapter.client = httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(500, text='fictional private error body')))
    try:
        with pytest.raises(TransportFailure) as error:
            adapter.infer(verified(out)['jobs'][0], recipes()['no_retrieval'])
        assert error.value.audit == {'error_type': 'HTTPStatusError', 'http_status': 500,
                                    'body': 'fictional private error body'}
    finally:
        adapter.close()
