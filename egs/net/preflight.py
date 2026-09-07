import csv
from dataclasses import replace
from pathlib import Path
import torch
from egs.backend import seed_for
from egs.checks import quality
from egs.io_utils import sha256_file, write_json, write_jsonl
from egs.protocol import protocol_id, role_lock
from . import METHODS
from .experiment import execute, measure



@torch.inference_mode()
def batch_numeric_probe(engine, question, generated_ids):
    """Compare conditional probabilities at IDENTICAL prefixes, before paths diverge.

    BF16 batched kernels need not round identically to batch=1. Whole-path token
    agreement is therefore a diagnostic, not a test of RNG stream independence.
    This local TV probe is not a bound on full answer-distribution distance.
    """
    probes = []
    positions = sorted({0, min(15, len(generated_ids) - 2), min(63, len(generated_ids) - 2)})
    for position in positions:
        if position < 0:
            continue
        probe_q = replace(question, input_ids=question.input_ids + generated_ids[:position])
        parent = engine.prefill(probe_q)
        token = generated_ids[position]
        probabilities = []
        for batch in [1, engine.cfg.backend.batch_size]:
            states = engine.fork(parent, batch)
            for state in states:
                state.ids = list(generated_ids[:position + 1])
            engine._forward(states, torch.full((batch,), token, dtype=torch.long, device=engine.device))
            logits = torch.cat([state.logits for state in states], 0)
            p = torch.softmax(logits.float(), -1).double()
            probabilities.append(p / p.sum(-1, keepdim=True))
        tv = float((probabilities[1] - probabilities[0]).abs().sum(-1).max() / 2)
        probes.append({'generated_position': position + 1, 'max_row_total_variation': tv,
                       'passed': tv <= engine.cfg.checks.batch_probability_tv_tolerance})
    return {'probes': probes, 'tolerance': engine.cfg.checks.batch_probability_tv_tolerance,
            'passed': bool(probes) and all(p['passed'] for p in probes),
            'scope': 'conditional next-token probabilities on fixed identical prefixes; not a full-path TV bound'}


def preflight(root, cfg, development, engine):
    root = Path(root)
    with role_lock(root, 'net-preflight'):
        if (root / 'preflight.json').exists():
            raise RuntimeError('preflight already exists; use a new run to repeat')
        records = []
        selected = development[:5]
        required = cfg.checks.manual_review_samples
        with torch.inference_mode():
            for i in range(0, required, 10):
                q = selected[(i // 10) % len(selected)]
                result = execute(engine, q, 'iid', min(10, required - i), seed_for(cfg.sampling.seed_base, 'preflight', i))
                for sample in result['samples']:
                    records.append({'review_id': str(len(records)), 'question_id': q.question_id,
                        'question': q.question, 'text': engine.decode(sample['token_ids']),
                        'answer': sample['answer'], 'finish_reason': sample['finish_reason'], **sample})
                print(f'preflight: {len(records)}/{required} development rollouts complete', flush=True)
        write_jsonl(root / 'preflight_rollouts.jsonl', records)
        with open(root / 'manual_review.csv', 'w', newline='', encoding='utf-8') as f:
            fields = ['review_id', 'question_id', 'question', 'text', 'parsed_answer', 'finish_reason', 'human_answer', 'approved', 'notes']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                writer.writerow({**{k: r[k] for k in fields if k in r}, 'parsed_answer': r['answer']})
        numerical = batch_numeric_probe(engine, selected[0], records[0]['token_ids'])
        comparisons = []
        for method in METHODS:
            print(f'preflight: {method} same-configuration replay and cross-batch diagnostic', flush=True)
            key = seed_for(cfg.sampling.seed_base, 'preflight', method)
            # This also checks every arithmetic/operator accounting path encountered.
            a = measure(engine, selected[0], method, cfg.study.iid_budgets[0], key)
            original_batch = cfg.backend.batch_size
            cfg.backend.batch_size = 1
            try:
                with torch.inference_mode():
                    b = execute(engine, selected[0], method, cfg.study.iid_budgets[0], key)
            finally:
                cfg.backend.batch_size = original_batch
            comparisons.append({'method': method, 'same_tokens_weights_different_batch': a['samples'] == b['samples'],
                'same_allocation': [x['children'] for x in a['allocation']] == [x['children'] for x in b['allocation']],
                'same_seed_same_configuration': a['cost']['instrumentation_replay_verified'],
                'single_prompt_prefill': a['ledger']['prompt_prefills'] == 1,
                'cost': a['cost']})
        diverse = any(len({tuple(r['token_ids']) for r in records if r['question_id'] == q.question_id}) > 1 for q in selected)
        rates = quality([r['answer'] for r in records], cfg)
        report = {'protocol_id': protocol_id(root), 'quality': rates, 'methods': comparisons,
                  'different_seed': diverse, 'data_scope': 'development only', 'batch_numerics': numerical,
                  'cross_batch_token_equality': 'diagnostic only; RNG indexing is separately tested',
                  'rollouts_sha256': sha256_file(root / 'preflight_rollouts.jsonl')}
        report['passed'] = rates['passed'] and diverse and numerical['passed'] and all(
            c['same_seed_same_configuration'] and c['single_prompt_prefill'] for c in comparisons)
        write_json(root / 'preflight.json', report)
        if not report['passed']:
            raise RuntimeError('stop-the-line: preflight failed; inspect preflight.json')
        return report
