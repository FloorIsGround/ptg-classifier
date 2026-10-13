"""Native transport boundary. No download, service management or remote execution."""
from __future__ import annotations

import ipaddress
import math
from urllib.parse import urlsplit

import httpx

from .contract import AdmissionBlocked, ContractError, digest, keys, require


SYSTEM_TEXT = ('Classify only the target; preceding turns are context, '
               'never substitute evidence for target attribution.')


class RuntimeDrift(ContractError):
    pass


class TransportFailure(ContractError):
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit or {'error_type': 'TransportFailure'}


def loopback(url):
    p = urlsplit(url)
    require(p.scheme == 'http' and p.hostname and p.port and not p.username and
            not p.password and not p.query and not p.fragment and p.path in ('', '/'),
            'bare numeric HTTP loopback origin with port required')
    require(ipaddress.ip_address(p.hostname).is_loopback, 'loopback only')
    return url.rstrip('/')


def probability_response(value, labels):
    """Vetted bridge envelope, not a claim about any server's actual wire format."""
    keys(value, 'labels noul')
    require(value['labels'] == labels, 'native probability question order differs')
    probs = value['noul']
    require(isinstance(probs, list) and len(probs) == len(labels), 'native vector length differs')
    require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probs),
            'native scores must be finite probabilities in [0,1]')
    return [int(p >= 0.5) for p in probs]


def validate_reply(value, job, recipe):
    if recipe['backend'] == 'd1_native':
        return probability_response(value, job['labels'])
    keys(value, 'model done done_reason message prompt_eval_count eval_count')
    if value['model'] != recipe['model']:
        raise RuntimeDrift('response model drift')
    require(value['done'] is True and value['done_reason'] == 'stop', 'incomplete generation')
    keys(value['message'], 'content thinking')
    require(not value['message']['thinking'], 'unexpected thinking')
    n = value['prompt_eval_count']
    require(type(n) is int and 0 <= n < recipe['decoding']['num_ctx'] -
            recipe['decoding']['num_predict'], 'possible prompt truncation')
    require(type(value['eval_count']) is int and 0 <= value['eval_count'] <=
            recipe['decoding']['num_predict'], 'invalid generated token count')
    raw = value['message']['content']
    require(isinstance(raw, str), 'invalid generated content')
    # Strict parser; no punctuation-stripping or explanation acceptance.
    parsed = {'yes': 1, 'no': 0}.get(raw.strip().casefold())
    return [parsed]


class SyntheticAdapter:
    """Strict deterministic fake for fixtures only; never a research inference result."""
    synthetic = True

    def __init__(self, runtime):
        self.runtime = runtime

    def identity(self):
        return self.runtime.copy()

    def infer(self, job, recipe):
        # Depends on target text/label, not gold. RAG/context changes remain observable.
        bits = [int(digest([job['state']['target'], label])[-1], 16) % 2 for label in job['labels']]
        if recipe['backend'] == 'd1_native':
            return {'labels': job['labels'], 'noul': [0.75 if b else 0.25 for b in bits]}
        return {'model': recipe['model'], 'done': True, 'done_reason': 'stop',
                'message': {'content': 'yes' if bits[0] else 'no', 'thinking': ''},
                'prompt_eval_count': len(job['wire'].encode('utf-8')), 'eval_count': 1}


class OllamaAdapter:
    synthetic = False

    def __init__(self, url, runtime):
        require(runtime['backend'] == 'ollama', 'Ollama runtime required')
        self.url = loopback(url)
        self.runtime = runtime
        self.client = httpx.Client(timeout=120, trust_env=False, follow_redirects=False)

    def close(self):
        self.client.close()

    def identity(self):
        version = self.client.get(self.url + '/api/version').raise_for_status().json()['version']
        model = self.runtime['model']
        shown = self.client.post(self.url + '/api/show', json={'model': model}).raise_for_status().json()
        models = self.client.get(self.url + '/api/tags').raise_for_status().json()['models']
        matches = [m for m in models if m.get('name') == model]
        if len(matches) != 1:
            raise RuntimeDrift('exact installed model missing')
        show = {k: shown.get(k) for k in ('template', 'parameters', 'modelfile', 'details',
                                         'model_info', 'capabilities')}
        return {**self.runtime, 'artifact_digest': matches[0]['digest'].removeprefix('sha256:'),
                'engine_version': version, 'show_digest': digest(show),
                'quantization': shown.get('details', {}).get('quantization_level')}

    def infer(self, job, recipe):
        payload = {'model': recipe['model'], 'messages': [
            {'role': 'system', 'content': SYSTEM_TEXT},
            {'role': 'user', 'content': job['wire']}], 'stream': False, 'think': False,
            'options': {k: v for k, v in recipe['decoding'].items() if k != 'think'}, 'keep_alive': '10m'}
        try:
            value = self.client.post(self.url + '/api/chat', json=payload)
            value.raise_for_status()
            try:
                data = value.json()
            except ValueError:
                return {'invalid_wire_text': value.text}
        except httpx.HTTPError as e:
            audit = {'error_type': type(e).__name__}
            if isinstance(e, httpx.HTTPStatusError):
                audit.update(http_status=e.response.status_code, body=e.response.text)
            raise TransportFailure(type(e).__name__, audit) from e
        # Keep complete wire response in private ledger; normalize supported fields for validation.
        if not isinstance(data, dict) or not isinstance(data.get('message'), dict):
            return {'invalid_wire_response': data}
        return {'wire_response': data, 'validated_response': {
            'model': data.get('model'), 'done': data.get('done'), 'done_reason': data.get('done_reason'),
            'message': {'content': data['message'].get('content'),
                        'thinking': data['message'].get('thinking', '')},
            'prompt_eval_count': data.get('prompt_eval_count'), 'eval_count': data.get('eval_count')}}


def admit(recipe, synthetic):
    if recipe['backend'] == 'granite_remote':
        raise AdmissionBlocked('Granite remote inference requires fresh authorization and a reviewed adapter')
    if recipe['backend'] == 'd1_native' and not synthetic:
        raise AdmissionBlocked('D1 needs a vetted system_one native backend, pinned weights and serializer; '
                               'chat generation and automatic trust_remote_code are prohibited')
