"""Deterministic causal input selection. Does not modify transformer attention."""
from .contract import keys, require, integer, text, labels_valid


def normalized(s):
    return ' '.join(s.casefold().split())


def validate_bundle(b):
    keys(b, 'schema synthetic labels codebook rows sources rankings retrieval_identity reviewed_blanks')
    require(b['schema'] == 'frozen-input/2' and type(b['synthetic']) is bool,
            'invalid frozen input schema')
    require(b['reviewed_blanks'] is True, 'reviewed blanks attestation required')
    labels_valid(b['labels'])
    require(isinstance(b['codebook'], list) and
            [e['label'] for e in b['codebook']] == b['labels'], 'codebook label order differs')
    for e in b['codebook']:
        keys(e, 'label definition positive negative')
        text(e['definition'])
        require(isinstance(e['positive'], list) and isinstance(e['negative'], list) and
                all(isinstance(s, str) for s in e['positive'] + e['negative']), 'invalid examples')
    require(isinstance(b['rows'], list) and b['rows'], 'rows required')
    ids, orders = set(), set()
    for row in b['rows']:
        keys(row, 'id semester meeting position text target gold coder_codes')
        require(row['semester'] in ('S26', 'SP26'), 'development S26/SP26 only')
        text(row['id']); text(row['meeting']); integer(row['position'])
        require(isinstance(row['text'], str) and type(row['target']) is bool, 'invalid row')
        order = (row['semester'], row['meeting'], row['position'])
        require(row['id'] not in ids and order not in orders, 'duplicate row/order key')
        ids.add(row['id']); orders.add(order)
        codes(row['gold'], b['labels'])
        require(isinstance(row['coder_codes'], list), 'coder cells required')
        for cell in row['coder_codes']:
            codes(cell, b['labels'])
        if row['target']:
            text(row['text'])
    targets = [r for r in b['rows'] if r['target']]
    require(targets, 'frozen targets required')
    meetings = {(r['semester'], r['meeting']) for r in b['rows']}
    target_texts = {normalized(r['text']) for r in targets}
    static = {normalized(s) for e in b['codebook'] for s in e['positive'] + e['negative']}
    require(not target_texts & static, 'static example target overlap')
    require(isinstance(b['sources'], list), 'source examples required')
    sources = {}
    for s in b['sources']:
        keys(s, 'id semester meeting text codes')
        text(s['id']); text(s['meeting']); text(s['text'])
        require(s['semester'] not in ('S26', 'SP26', 'FA26'), 'development/held-out retrieval source')
        require((s['semester'], s['meeting']) not in meetings and
                normalized(s['text']) not in target_texts, 'source target overlap')
        codes(s['codes'], b['labels'])
        require(s['id'] not in sources and s['id'] not in ids, 'duplicate source id')
        sources[s['id']] = s
    require(isinstance(b['rankings'], dict) and
            set(b['rankings']) == {r['id'] for r in targets}, 'frozen ranking keys differ')
    for ranking in b['rankings'].values():
        require(isinstance(ranking, list) and len(set(ranking)) == len(ranking) and
                all(i in sources for i in ranking), 'invalid source ranking')
    keys(b['retrieval_identity'], 'algorithm index_revision embedding_revision ranking_revision')
    for v in b['retrieval_identity'].values():
        text(v)
    return targets, sources


def codes(cell, labels):
    # Empty individual-coder cells are observed no-code decisions, never uncertainty.
    if cell == '':
        return []
    require(isinstance(cell, list) and len(set(cell)) == len(cell) and
            all(c in labels for c in cell), 'invalid/unknown codes; blank must be empty string')
    return cell


def context_for(rows, target, policy):
    prior = sorted((r for r in rows if r['semester'] == target['semester'] and
                    r['meeting'] == target['meeting'] and r['position'] < target['position']),
                   key=lambda r: r['position'], reverse=True)[:policy['predecessors']]
    chosen, total = [], 0
    for distance, row in enumerate(prior):
        value = row['text'].strip()
        if not value:
            continue
        cost = len(value.encode('utf-8')) + 1
        # Exact integer weighted byte cost doubles with predecessor distance.
        # Select a contiguous recent prefix of whole nonblank turns; stop at first overflow.
        weighted = cost if policy['policy'] == 'fixed_predecessors' else cost * (2 ** distance)
        if total + weighted > policy['budget_bytes']:
            require(policy['policy'] != 'fixed_predecessors', 'fixed context budget overflow')
            break
        total += weighted
        chosen.append(row)
    return [r['text'].strip() for r in reversed(chosen)]


def retrieved_for(bundle, sources, target, label, policy):
    if policy['policy'] == 'none':
        return []
    return [sources[i]['text'] for i in bundle['rankings'][target['id']][:policy['top_k']]
            if label in codes(sources[i]['codes'], bundle['labels'])]
