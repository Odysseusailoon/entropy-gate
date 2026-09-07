"""Predetermined complete runs, proper root weights, end-to-end measurement."""
from __future__ import annotations
import copy
import time
import torch
from egs.answer_parser import parse_answer
from egs.backend import seed_for
from egs.io_utils import sha256_obj
from egs.metrics import TRUNCATED
from . import METHODS
from .arithmetic import Sampler
from .flops import Census


def allocate(scores, total, minimum, beta, uniform=False):
    n = len(scores)
    if total < n * minimum:
        raise ValueError('infeasible branch allocation')
    logits = torch.zeros_like(scores) if uniform else scores * beta
    ideal = (total - n * minimum) * torch.softmax(logits, 0)
    whole = torch.floor(ideal).to(torch.int64)
    remaining = total - n * minimum - int(whole.sum())
    order = torch.argsort(ideal - whole, descending=True, stable=True)
    whole[order[:remaining]] += 1
    return (whole + minimum).tolist()


def answer_distribution(engine, states, weights):
    answers = [TRUNCATED if s.reason != 'eos' else parse_answer(engine.decode(s.ids)).answer for s in states]
    keys = sorted(set(answers))
    mapping = {key: i for i, key in enumerate(keys)}
    indices = torch.tensor([mapping[a] for a in answers], dtype=torch.long, device=engine.device)
    mass = torch.zeros(len(keys), dtype=torch.float64, device=engine.device)
    mass.scatter_add_(0, indices, weights)
    if not bool(torch.isclose(mass.sum(), torch.ones((), dtype=torch.float64, device=engine.device), atol=1e-10, rtol=0)):
        raise RuntimeError('tree total weight must equal one')
    return dict(zip(keys, mass.tolist())), answers


def execute(engine, question, method, samples, seed):
    """No reference access and no outcome-dependent sample-count stopping."""
    if method not in METHODS:
        raise ValueError(method)
    cfg = engine.cfg
    engine.reset_ledger()
    root = engine.prefill(question)
    allocation = []
    if method in {'iid', 'arithmetic'}:
        paths = engine.fork(root, samples)
        sampler = Sampler(method, samples, seed, engine.device, cfg.study.arithmetic_permute_vocab, max_tokens=cfg.sampling.max_new_tokens)
        engine.advance(paths, sampler)
        weights = torch.ones(samples, dtype=torch.float64, device=engine.device) / samples
        root_ids = list(range(samples)) if method == 'iid' else [0] * samples
    else:
        r = cfg.study.roots  # fixed BEFORE seeing any generated prefix
        prefixes = engine.fork(root, r)
        prefix_seed = seed_for(seed, 'prefixes')
        sampler = Sampler('iid', r, prefix_seed, engine.device)
        engine.advance(prefixes, sampler, stop_at=cfg.study.prefix_tokens, record_entropy=method == 'entropy')
        scores = []
        for prefix in prefixes:
            hs = [h for h in prefix.entropies if h is not None]
            scores.append(torch.stack(hs).mean() if hs else torch.zeros((), device=engine.device))
        scores = torch.stack(scores)
        counts = allocate(scores, samples, cfg.study.n_min, cfg.study.beta, uniform=method == 'uniform')
        paths, root_ids, path_seeds, weights_list = [], [], [], []
        for i, (prefix, k) in enumerate(zip(prefixes, counts)):
            allocation.append({'root': i, 'prefix_ids': list(prefix.ids), 'prefix_length': len(prefix.ids),
                               'children': k, 'score': float(scores[i]), 'prefix_finish': prefix.reason,
                               'available_information': 'generated prefix only', 'parent_weight': 1 / r})
            paths.extend(engine.fork(prefix, k))
            root_ids.extend([i] * k)
            path_seeds.extend(seed_for(seed, 'child', i, j) for j in range(k))
            weights_list.append(torch.ones(k, dtype=torch.float64, device=engine.device) / r / k)
        weights = torch.cat(weights_list)
        continuation_sampler = Sampler('iid', len(paths), seed_for(seed, 'continuations'), engine.device, path_seeds=path_seeds)
        engine.advance(paths, continuation_sampler)
    distribution, answers = answer_distribution(engine, paths, weights)
    samples_out = [{'root': rid, 'token_ids': state.ids, 'finish_reason': state.reason, 'answer': answer,
                    'weight': weight} for rid, state, answer, weight in zip(root_ids, paths, answers, weights.tolist())]
    return {'method': method, 'samples_requested': samples, 'seed': seed,
            'arithmetic_precision': ({'shift_bits': sampler.codes.bits, 'exact_residuals': True,
                'boundary_corrections': sampler.codes.corrections} if method == 'arithmetic' else None),
            'distribution': distribution, 'samples': samples_out, 'allocation': allocation,
            'ledger': dict(engine.ledger), 'pilot_policy': 'no discarded test pilot; fixed prefixes are retained',
            'synthetic': engine.synthetic}


def signature(result):
    # Gate's floating-point scores may differ by reduction rounding. Their decisions
    # and all sample tokens/weights must nevertheless replay exactly.
    return sha256_obj({'samples': result['samples'], 'counts': [a['children'] for a in result['allocation']],
                       'ledger': result['ledger']})


def measure(engine, question, method, samples, seed):
    """Primary GPU seconds excludes instrumentation; exact-seed replay counts FLOPs.

    The measured interval includes prompt prefill, cache copies, generation,
    entropy, allocation, Python scheduling, parsing and weighted aggregation.
    Profiling replay is recorded separately as measurement overhead.
    """
    engine.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = execute(engine, question, method, samples, seed)
    engine.synchronize()
    elapsed = time.perf_counter() - start
    engine.synchronize()
    begin_profile = time.perf_counter()
    with torch.inference_mode(), Census() as counter:
        replay = execute(engine, question, method, samples, seed)
    engine.synchronize()
    profiling_elapsed = time.perf_counter() - begin_profile
    counter.require_coverage()
    if signature(result) != signature(replay):
        raise RuntimeError('instrumented replay changed tokens, allocation, weights or execution shape')
    result['cost'] = {**counter.report(), 'gpu_seconds': elapsed if engine.is_cuda else None,
                      'wall_seconds': elapsed, 'profiling_replay_seconds': profiling_elapsed,
                      'total_measurement_wall_seconds': elapsed + profiling_elapsed,
                      'time_scope': 'one GPU occupied by complete algorithm; synchronized wall time',
                      'instrumentation_replay_verified': True}
    return result
