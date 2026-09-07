"""Independent IID references, validated imports and bounded size extensions."""
from pathlib import Path
from collections import Counter
import numpy as np
from egs.io_utils import read_json, read_jsonl, sha256_file, sha256_obj, write_json, write_jsonl
from egs.metrics import histogram, js_divergence, INVALID, TRUNCATED
from egs.protocol import protocol_id, role_lock
from egs.io_utils import write_csv
from .pipeline import load_result, save_result
from .protocol import target_signature
from .config import load_config


def records_for(root, qid):
    root = Path(root)
    imported = root / 'reference_import' / f'{qid}.json'
    records, start = [], 0
    if imported.exists():
        value = load_result(imported)
        records = value['samples']
        start = len(records)
    for path in sorted((root / 'reference_chunks' / qid).glob('*.json')):
        value = load_result(path)
        offset = value['start']
        if offset < start:
            if imported.exists():
                continue
            raise RuntimeError('overlapping reference chunks')
        if offset != start:
            raise RuntimeError('missing reference chunk')
        records.extend(value['samples'])
        start += len(value['samples'])
    return records


def seal_reference(root, cfg, questions):
    root = Path(root)
    sealpath = root / 'reference_complete.json'
    if sealpath.exists():
        seal = read_json(sealpath)
        if seal['protocol_id'] != protocol_id(root) or seal['sha256'] != sha256_file(root / 'reference_distributions.jsonl'):
            raise RuntimeError('sealed reference was modified')
        return seal
    with role_lock(root, 'reference-seal'):
        distributions, convergence, all_invalid, all_truncated = [], [], [], []
        comparisons, input_hashes = [], {}
        for q in questions:
            rows = records_for(root, q.question_id)
            if len(rows) < cfg.reference.n_ref:
                raise RuntimeError(f'independent reference incomplete for {q.question_id}')
            answers = [s['answer'] for s in rows]
            dist = histogram(answers)
            n = len(rows)
            grid = sorted(set([g for g in cfg.reference.convergence_grid if g <= n] + [n // 2, n]))
            for k in grid:
                error = js_divergence(histogram(answers[:k]), dist)
                convergence.append({'question_id': q.question_id, 'n': k, 'n_full': n, 'js_bits': error})
                if k == n // 2:
                    comparisons.append(error)
            invalid, truncated = dist.get(INVALID, 0), dist.get(TRUNCATED, 0)
            all_invalid.append(invalid)
            all_truncated.append(truncated)
            distributions.append({'question_id': q.question_id, 'n': n, 'distribution': dist,
                                  'histogram': dict(Counter(answers)), 'prompt_hash': sha256_obj(q.input_ids)})
            sourcepaths = list((root / 'reference_chunks' / q.question_id).glob('*.json'))
            imported = root / 'reference_import' / f'{q.question_id}.json'
            if imported.exists():
                sourcepaths.append(imported)
            input_hashes.update({str(p.relative_to(root)): sha256_file(p) for p in sourcepaths})
        stable = float(np.mean(comparisons)) <= cfg.reference.convergence_js_threshold
        quality = np.mean(all_invalid) < cfg.checks.invalid_threshold and np.mean(all_truncated) <= cfg.checks.truncation_threshold
        report = {'protocol_id': protocol_id(root), 'mean_half_vs_full_js': float(np.mean(comparisons)),
                  'max_question_half_vs_full_js': max(comparisons), 'passed': bool(stable and quality),
                  'invalid_rate': float(np.mean(all_invalid)), 'truncation_rate': float(np.mean(all_truncated)),
                  'n_by_question': {r['question_id']: r['n'] for r in distributions},
                  'threshold_js': cfg.reference.convergence_js_threshold}
        write_json(root / 'reference_checks.json', report)
        write_csv(root / 'reference_convergence.csv', convergence)
        if not quality:
            raise RuntimeError('reference invalid/truncation quality gate failed')
        if not stable:
            raise RuntimeError('reference not stable: extend independent reference before scoring')
        write_jsonl(root / 'reference_distributions.jsonl', distributions)
        seal = {**report, 'sha256': sha256_file(root / 'reference_distributions.jsonl'), 'input_hashes': input_hashes}
        write_json(sealpath, seal)
        return seal


def import_reference(root, cfg, questions, fingerprint, source):
    """Reuse source raw IID outcomes only after target/prompt/independence checks."""
    root, source = Path(root), Path(source)
    if root.resolve() == source.resolve() or (root / 'queue.sqlite').exists():
        raise RuntimeError('import an independent reference before initializing queues')
    source_manifest = read_json(source / 'manifest.json')
    if source_manifest.get('protocol') != 'net-benefit-v3':
        raise RuntimeError('reference import requires a sealed v3 source')
    source_cfg = load_config(source / 'config.yaml')
    source_fp = read_json(source / 'prompt_fingerprint.json')
    if target_signature(cfg, fingerprint) != target_signature(source_cfg, source_fp):
        raise RuntimeError('reference target mismatch (model, tokenizer, parser, dtype, sampling or EOS)')
    # Same seed namespace can collide across reused/new runs. Require a different
    # base seed; seeds are cheap, independence is essential.
    if source_cfg.sampling.seed_base == cfg.sampling.seed_base:
        raise RuntimeError('reference import requires a different source seed_base')
    versions = source_manifest.get('runtime_versions', {})
    from egs.protocol import runtime_versions
    current = runtime_versions()
    for name in ['torch', 'transformers', 'tokenizers']:
        if versions.get(name) != current.get(name):
            raise RuntimeError(f'reference numerical runtime differs: {name}')
    source_qs = {q['question_id']: q for q in read_jsonl(source / 'questions.jsonl')}
    seal = read_json(source / 'reference_complete.json')
    expected = seal.get('sha256')
    if not seal.get('passed', True) or expected != sha256_file(source / 'reference_distributions.jsonl'):
        raise RuntimeError('source reference seal invalid')
    if seal.get('protocol_id') != protocol_id(source):
        raise RuntimeError('source reference protocol seal invalid')
    for relative, expected_hash in seal.get('input_hashes', {}).items():
        if sha256_file(source / relative) != expected_hash:
            raise RuntimeError('source raw reference was modified after sealing')
    prepared = []
    for q in questions:
        if q.question_id not in source_qs or source_qs[q.question_id]['input_ids'] != q.input_ids:
            raise RuntimeError(f'reference has no identical prompt for {q.question_id}')
        sources = list((source / 'reference_chunks' / q.question_id).glob('*.json'))
        imported = source / 'reference_import' / f'{q.question_id}.json'
        if imported.exists():
            sources.append(imported)
        if not sources or any(str(p.relative_to(source)) not in seal.get('input_hashes', {}) for p in sources):
            raise RuntimeError('source reference contains unsealed raw samples')
        rows = records_for(source, q.question_id)
        if len(rows) < cfg.reference.n_ref:
            raise RuntimeError(f'reference too small for {q.question_id}: {len(rows)} < {cfg.reference.n_ref}')
        prepared.append((q, rows))
    for q, rows in prepared:
        save_result(root / 'reference_import' / f'{q.question_id}.json', {'samples': rows, 'source': str(source.resolve()),
                    'source_manifest_sha256': sha256_file(source / 'manifest.json'), 'target_signature': target_signature(cfg, fingerprint)})
    return seal_reference(root, cfg, questions)


def initialize_extension(root, cfg, questions, samples):
    from .queue import Queue
    from .pipeline import read_plan
    read_plan(root)
    queue = Queue(root)
    try:
        queue.require_complete('test')
        if queue.db.execute("SELECT 1 FROM tasks WHERE phase LIKE 'reference-%' AND state != 'done' LIMIT 1").fetchone():
            raise RuntimeError('finish the existing reference extension first')
    finally:
        queue.close()
    if (Path(root) / 'reference_complete.json').exists():
        raise RuntimeError('accepted reference is immutable')
    allowed, n = [], cfg.reference.n_ref
    while n <= cfg.study.reference_max_samples:
        allowed.append(n)
        n *= 2
    if samples not in allowed[1:]:
        raise ValueError(f'extension size must be a preregistered doubling: {allowed[1:]}')
    jobs = []
    for q in questions:
        start = len(records_for(root, q.question_id))
        if start >= samples:
            continue
        for lo in range(start, samples, 128):
            jobs.append({'kind': 'reference', 'question_id': q.question_id, 'start': lo,
                         'samples': min(128, samples - lo)})
    if not jobs:
        raise RuntimeError('reference already has requested sample count')
    phase = f'reference-{samples}'
    queue = Queue(root)
    queue.initialize(phase, jobs, sha256_obj([protocol_id(root), samples]), cfg.study.queue_seed + samples)
    queue.close()
    return phase
