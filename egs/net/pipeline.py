"""Development calibration -> sealed plan -> shuffled paired test blocks."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import os
import random
import socket
import time
import numpy as np
import torch
from egs.backend import seed_for
from egs.io_utils import read_json, sha256_file, sha256_obj, write_json
from egs.metrics import INVALID, TRUNCATED
from egs.protocol import protocol_id, require_ready, role_lock
from . import METHODS
from .experiment import execute, measure
from .queue import Queue


def ordered_methods(seed):
    order = list(METHODS)
    random.Random(seed).shuffle(order)
    return order


def plan_id(root):
    return sha256_file(Path(root) / 'calibration_plan.json')


def read_plan(root):
    root = Path(root)
    path = root / 'calibration_plan.json'
    seal = read_json(root / 'calibration_seal.json')
    if sha256_file(path) != seal['plan_sha256'] or seal['protocol_id'] != protocol_id(root):
        raise RuntimeError('calibration plan changed or is from another protocol')
    return read_json(path)


def initialize(root, cfg, test, dev, phase):
    require_ready(root, cfg)
    queue = Queue(root)
    jobs = []
    if phase == 'calibration':
        for q in dev:
            for n in cfg.study.candidates:
                for rep in range(cfg.study.dev_replicates):
                    key = seed_for(cfg.study.queue_seed, phase, q.question_id, n, rep)
                    jobs.append({'kind': 'methods', 'question_id': q.question_id, 'budget': n, 'replicate': rep,
                                 'order': ordered_methods(key)})
        fingerprint = protocol_id(root)
    elif phase == 'test':
        read_plan(root)
        for q in test:
            for b in cfg.study.iid_budgets:
                for rep in range(cfg.study.replicates):
                    key = seed_for(cfg.study.queue_seed, phase, q.question_id, b, rep)
                    jobs.append({'kind': 'methods', 'question_id': q.question_id, 'budget': b, 'replicate': rep,
                                 'order': ordered_methods(key)})
            # Reference jobs share the SAME queue and workers; chunking avoids a
            # dedicated reference GPU and a long tail from one giant question job.
            if not (Path(root) / 'reference_import' / f'{q.question_id}.json').exists():
                for start in range(0, cfg.reference.n_ref, 128):
                    jobs.append({'kind': 'reference', 'question_id': q.question_id, 'start': start,
                                 'samples': min(128, cfg.reference.n_ref - start)})
        fingerprint = sha256_obj([protocol_id(root), plan_id(root)])
    else:
        raise ValueError(phase)
    queue.initialize(phase, jobs, fingerprint, cfg.study.queue_seed)
    result = queue.status(phase)
    queue.close()
    return result


def result_path(root, phase, qid, budget, rep, method):
    return Path(root) / 'results' / phase / f'{qid}_b{budget}_r{rep}_{method}.json'


def save_result(path, value):
    write_json(path, {'sha256': sha256_obj(value), 'result': value})


def load_result(path):
    checkpoint = read_json(path)
    if checkpoint['sha256'] != sha256_obj(checkpoint['result']):
        raise RuntimeError(f'corrupt result checkpoint: {path}')
    return checkpoint['result']


def hardware(engine):
    if engine.synthetic:
        return {'name': 'synthetic-cpu', 'cuda': False, 'hostname': socket.gethostname()}
    if not engine.is_cuda:
        raise RuntimeError('real queue workers require CUDA; CPU is for unit tests only')
    prop = torch.cuda.get_device_properties(engine.device)
    required = engine.cfg.study.required_gpu_name
    if required not in prop.name:
        raise RuntimeError(f'this frozen study requires {required}, found {prop.name}')
    return {'name': prop.name, 'cuda': True, 'capability': [prop.major, prop.minor],
            'memory_bytes': prop.total_memory, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'hostname': socket.gethostname()}


def worker(root, cfg, test, dev, engine, phase, worker_id):
    root = Path(root)
    require_ready(root, cfg)
    plan = read_plan(root) if phase == 'test' else None
    queue = Queue(root)
    status = queue.status(phase)
    if status.get('failed'):
        queue.close()
        raise RuntimeError('queue contains a failed block; do not spend more GPU time before resolving it')
    if not status.get('pending') and not status.get('running'):
        queue.close()
        return status
    info = hardware(engine)
    compatibility = sha256_obj({k: v for k, v in info.items() if k not in {'hostname', 'visible_devices'}})
    with queue.transaction():
        old = queue.db.execute("SELECT value FROM metadata WHERE key='hardware' ").fetchone()
        if old and old[0] != compatibility:
            raise RuntimeError('mixed hardware in one calibrated study')
        if not old:
            queue.db.execute("INSERT INTO metadata VALUES('hardware',?)", (compatibility,))
    # Always warm every method. All warmup cost is archived and amortized in the
    # primary net-cost analysis, including discarded warmup answers.
    warmups = []
    for repeat in range(cfg.study.warmup_rollouts):
        for method in METHODS:
            result = measure(engine, dev[0], method, cfg.study.roots * cfg.study.n_min,
                             seed_for(cfg.sampling.seed_base, 'warmup', phase, worker_id, repeat, method))
            warmups.append({'method': method, 'cost': result['cost']})
    stamp = f'{phase}_{worker_id}_{os.getpid()}_{time.time_ns()}'
    write_json(root / 'worker_overhead' / f'{stamp}.json', {'worker': worker_id, 'phase': phase, 'hardware': info,
                                                        'warmups': warmups})
    questions = {q.question_id: q for q in (dev if phase == 'calibration' else test)}
    while True:
        task = queue.claim(phase, worker_id)
        if task is None:
            break
        try:
            q = questions[task['question_id']]
            if task['kind'] == 'reference':
                path = root / 'reference_chunks' / q.question_id / f"{task['start']:06d}.json"
                if not path.exists():
                    seed = seed_for(cfg.sampling.seed_base, 'net-independent-reference', q.question_id, task['start'])
                    engine.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        result = execute(engine, q, 'iid', task['samples'], seed)
                    engine.synchronize()
                    result.update({'question_id': q.question_id, 'start': task['start'],
                                   'evaluation_seconds': time.perf_counter() - start, 'protocol_id': protocol_id(root)})
                    save_result(path, result)
                else:
                    load_result(path)
            else:
                b, rep = task['budget'], task['replicate']
                for method in task['order']:
                    path = result_path(root, phase, q.question_id, b, rep, method)
                    n = b if phase == 'calibration' else plan['sample_counts'][method][str(b)]
                    # Uniform/entropy share root and child RNG streams for pairing.
                    family = 'fork' if method in {'uniform', 'entropy'} else method
                    seed = seed_for(cfg.sampling.seed_base, 'net', phase, q.question_id, b, rep, family)
                    if path.exists():
                        result = load_result(path)
                        if result['seed'] != seed or result['samples_requested'] != n:
                            raise RuntimeError('resumed run differs from frozen plan')
                        continue
                    result = measure(engine, q, method, n, seed)
                    result.update({'question_id': q.question_id, 'budget': b, 'replicate': rep,
                        'phase': phase, 'protocol_id': protocol_id(root), 'plan_id': plan_id(root) if plan else None,
                        'worker': worker_id, 'hardware': info, 'method_order': task['order']})
                    save_result(path, result)
            queue.finish(task['id'], worker_id)
            print(f"{worker_id}: {phase} {task['kind']} {q.question_id} completed", flush=True)
        except BaseException as exc:
            queue.finish(task['id'], worker_id, f'{type(exc).__name__}: {exc}')
            raise
    result = queue.status(phase)
    queue.close()
    return result


def warmup_costs(root, phases):
    costs = {m: {'total_flops': 0., 'gpu_seconds': 0., 'wall_seconds': 0.} for m in METHODS}
    for path in (Path(root) / 'worker_overhead').glob('*.json'):
        row = read_json(path)
        if row['phase'] not in phases:
            continue
        for entry in row['warmups']:
            for key in costs[entry['method']]:
                value = entry['cost'][key]
                if value is None:
                    costs[entry['method']][key] = None
                elif costs[entry['method']][key] is not None:
                    costs[entry['method']][key] += value
    return costs


def calibrate(root, cfg, test, dev):
    root = Path(root)
    with role_lock(root, 'calibration-seal'):
        if (root / 'calibration_seal.json').exists():
            return read_plan(root)
        queue = Queue(root)
        queue.require_complete('calibration')
        queue.close()
        values, sums, checksums = defaultdict(list), warmup_costs(root, {'calibration'}), {}
        quality = {m: {'invalid': [], 'truncated': []} for m in METHODS}
        for q in dev:
            for n in cfg.study.candidates:
                for rep in range(cfg.study.dev_replicates):
                    for method in METHODS:
                        path = result_path(root, 'calibration', q.question_id, n, rep, method)
                        r = load_result(path)
                        if r['protocol_id'] != protocol_id(root) or not r['cost']['coverage_passed']:
                            raise RuntimeError('invalid calibration record')
                        checksums[str(path.relative_to(root))] = sha256_file(path)
                        quality[method]['invalid'].append(r['distribution'].get(INVALID, 0.))
                        quality[method]['truncated'].append(r['distribution'].get(TRUNCATED, 0.))
                        values[(method, n)].append(r['cost']['total_flops'])
                        for key in sums[method]:
                            value = r['cost'][key]
                            if value is None:
                                sums[method][key] = None
                            elif sums[method][key] is not None:
                                sums[method][key] += value
        # A clean preflight does not guarantee that later development paths end
        # normally. Never seal budgets from a calibration that fails the same
        # weighted quality rules as the formal analysis; preserve its raw costs.
        quality_report = {}
        for method, rates in quality.items():
            invalid, truncated = float(np.mean(rates['invalid'])), float(np.mean(rates['truncated']))
            if invalid >= cfg.checks.invalid_threshold or truncated > cfg.checks.truncation_threshold:
                raise RuntimeError(f'stop-the-line: {method} calibration invalid/truncation rate exceeded the frozen threshold; no budget plan sealed')
            quality_report[method] = {'invalid_rate': invalid, 'truncation_rate': truncated,
                                      'independent_runs': len(rates['invalid'])}
        jobs_per_method = len(test) * len(cfg.study.iid_budgets) * cfg.study.replicates
        shares = {m: {k: v / jobs_per_method if v is not None else None for k, v in row.items()} for m, row in sums.items()}
        counts, matches = {m: {} for m in METHODS}, []
        for b in cfg.study.iid_budgets:
            target = float(np.mean(values[('iid', b)])) + shares['iid']['total_flops']
            for method in METHODS:
                choices = [b] if method == 'iid' else cfg.study.candidates
                selected = min(choices, key=lambda n: (abs(float(np.mean(values[(method, n)])) + shares[method]['total_flops'] - target), n))
                actual = float(np.mean(values[(method, selected)])) + shares[method]['total_flops']
                relative = abs(actual - target) / target
                if relative > cfg.study.calibration_max_relative_error:
                    raise RuntimeError(f'{method} cannot match development budget {b} within frozen tolerance; expand candidate grid in a new run')
                counts[method][str(b)] = selected
                matches.append({'method': method, 'iid_anchor': b, 'samples': selected,
                                'target_flops': target, 'development_flops': actual, 'relative_gap': relative})
        plan = {'protocol_id': protocol_id(root), 'sample_counts': counts, 'matches': matches,
                'development_quality': quality_report,
                'development_result_hashes': checksums, 'development_ids': [q.question_id for q in dev],
                'held_out_ids': [q.question_id for q in test], 'setup_share_per_test_run': shares,
                'setup_total': sums, 'jobs_per_method': jobs_per_method,
                'calibration_axis': 'total_flops', 'selection_uses': 'development costs only; no test/reference outcomes',
                'roots': cfg.study.roots, 'prefix_tokens': cfg.study.prefix_tokens, 'beta': cfg.study.beta,
                'test_rule': 'finish all preselected paths; no time/token cutoff based on observed cost'}
        write_json(root / 'calibration_plan.json', plan)
        write_json(root / 'calibration_seal.json', {'protocol_id': protocol_id(root), 'plan_sha256': plan_id(root)})
        return plan
