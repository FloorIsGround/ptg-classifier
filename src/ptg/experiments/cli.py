"""Synthetic-first CLI. Outputs are allowlisted counts or generic errors only."""
from __future__ import annotations

import argparse

import httpx
import json
from pathlib import Path

from .adapters import OllamaAdapter, SyntheticAdapter
from .analysis import export_sources, public_metadata, report
from .contract import ContractError, ROOT, recipes, require
from .suite import prepare, run, verified


def synthetic_bundle():
    labels = ['Fictional label %02d' % n for n in range(1, 22)]
    rows = [{'id': 'fictional-' + str(n), 'semester': 'S26' if n < 3 else 'SP26',
             'meeting': 'fictional-meeting', 'position': n % 3,
             'text': 'Fictional target turn %d about a pretend design.' % n,
             'target': n % 3 > 0, 'gold': [labels[n % 21]] if n % 2 else '',
             'coder_codes': ['', [labels[n % 21]]]} for n in range(6)]
    return {'schema': 'frozen-input/2', 'synthetic': True, 'labels': labels,
            'codebook': [{'label': l, 'definition': 'Fictional behavior category ' + l,
                          'positive': ['Fictional static positive ' + l],
                          'negative': ['Fictional static boundary ' + l]} for l in labels],
            'rows': rows, 'sources': [{'id': 'fictional-source', 'semester': 'F25',
                'meeting': 'fictional-training', 'text': 'Fictional previously coded source example.',
                'codes': labels}], 'rankings': {r['id']: ['fictional-source'] for r in rows if r['target']},
            'retrieval_identity': {'algorithm': 'offline-frozen-global-top5-label-filter',
                'index_revision': 'fictional-index-v1', 'embedding_revision': 'fictional-embedding-v1',
                'ranking_revision': 'fictional-rankings-v1'}, 'reviewed_blanks': True}


def synthetic_runtime(recipe):
    return {'backend': recipe['backend'], 'model': recipe['model'], 'artifact_digest': '0' * 64,
            'engine_version': 'fictional-engine-v1', 'show_digest': '1' * 64,
            'quantization': 'fictional-none', 'token_bound': 'utf8_bytes_upper_bound_v1',
            'template_overhead_tokens': 512, 'hardware_identity': 'fictional-cpu'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('list'); sub.add_parser('validate')
    for command in ('prepare', 'dry-run'):
        c = sub.add_parser(command)
        c.add_argument('--recipes', nargs='+', required=True)
        c.add_argument('--synthetic', action='store_true')
        c.add_argument('--input', type=Path)
        c.add_argument('--bindings', type=Path)
        c.add_argument('--factorial', action='store_true')
        if command == 'prepare':
            c.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('run')
    c.add_argument('--out', type=Path, required=True); c.add_argument('--repeat', required=True)
    c.add_argument('--resume', action='store_true'); c.add_argument('--synthetic', action='store_true')
    c.add_argument('--url')
    c = sub.add_parser('report')
    c.add_argument('--out', type=Path, required=True); c.add_argument('--repeats', nargs='+', required=True)
    c = sub.add_parser('export-public')
    c.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('export-sources')
    c.add_argument('--include', action='append', required=True); c.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command in ('list', 'validate'):
            result = {'schema': 'classifier-suite/2.0', 'recipes': list(recipes()), 'validated': True}
        elif args.command in ('prepare', 'dry-run'):
            require(args.synthetic != bool(args.input), 'choose synthetic or explicit private input')
            require(bool(args.bindings) != args.synthetic, 'private input requires explicit runtime bindings')
            registry = recipes()
            bundle = synthetic_bundle() if args.synthetic else json.loads(args.input.read_text())
            require(bundle['synthetic'] == args.synthetic, 'input synthetic attestation differs')
            bindings = {n: synthetic_runtime(registry[n]) for n in args.recipes} if args.synthetic else json.loads(args.bindings.read_text())
            plan = prepare(bundle, args.recipes, bindings, getattr(args, 'out', None), args.factorial)
            result = {'prepared_jobs': len(plan['jobs']), 'target_rows': sum(r['target'] for r in bundle['rows']),
                      'labels': len(bundle['labels']), 'arms': len(plan['arms']), 'synthetic': args.synthetic}
        elif args.command == 'run':
            plan = verified(args.out)
            runtime = next(iter(plan['arms'].values()))['classifier']['runtime']
            require(args.synthetic != bool(args.url), 'choose synthetic or explicit loopback URL')
            adapter = SyntheticAdapter(runtime) if args.synthetic else OllamaAdapter(args.url, runtime)
            try:
                result = run(args.out, args.repeat, adapter, args.resume)
            finally:
                if not args.synthetic:
                    adapter.close()
        elif args.command == 'report':
            result = report(args.out, args.repeats)
        elif args.command == 'export-public':
            result = public_metadata(verified(args.out))
        else:
            result = export_sources(args.include, args.out)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ContractError, ValueError, KeyError, OSError, TypeError, httpx.HTTPError):
        # No private paths, IDs, payloads, remote errors, or hashes in CLI errors.
        print(json.dumps({'error': 'validation/admission failed; inspect contract and private artifacts'}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
