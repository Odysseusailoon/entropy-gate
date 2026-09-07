import math
from fractions import Fraction
import numpy as np
import pytest
import torch
from egs.data import build_questions, Question
from egs.net.config import load_config
from egs.net.engine import Engine
from egs.net.experiment import execute, measure
from egs.net.arithmetic import ExactCodes, inverse_cdf, Sampler
from egs.net.flops import Census
from egs.net.analysis import bootstrap, speedup, squared_l2

@pytest.fixture
def toy():
    cfg = load_config('configs/net-demo.yaml')
    qs, fp = build_questions(cfg)
    return cfg, Engine(cfg, fp), qs[0]


def test_arithmetic_interval_and_shared_shift():
    codes = ExactCodes(3, 19, 128, 102)
    # The offsets between initial codes are a shared translated i/(N+1) lattice.
    assert (codes.values[1] - codes.values[0]) % 1 == Fraction(1, 4)
    p = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    token, remainder = inverse_cdf(p, torch.tensor([.35], dtype=torch.float64))
    assert token.item() == 1
    assert remainder.item() == pytest.approx(.5)
    # Unlike plain double recursion, residual precision survives >53 fair bits.
    initial = codes.values[0]
    cdf = torch.tensor([[.5, 1.]], dtype=torch.float64)
    for _ in range(100):
        guess = torch.searchsorted(cdf, codes.floats([0], 'cpu')[:, None], right=True).squeeze(1)
        codes.update(cdf, guess, [0])
    assert codes.values[0] == (initial * 2**100) % 1
    assert codes.values[0] != 0


def test_all_methods_one_prompt_and_cost_coverage(toy):
    cfg, engine, q = toy
    results = {m: measure(engine, q, m, 16, 104) for m in ['iid', 'uniform', 'entropy', 'arithmetic']}
    for r in results.values():
        assert r['ledger']['prompt_prefills'] == 1
        assert r['ledger']['prefill_tokens'] == len(q.input_ids)
        assert r['cost']['coverage_passed'] and r['cost']['total_flops'] > 0
        assert r['cost']['gpu_seconds'] is None  # no fabricated GPU timings
        assert sum(s['weight'] for s in r['samples']) == pytest.approx(1)
    assert results['uniform']['ledger']['decode_forward_tokens'] < results['iid']['ledger']['decode_forward_tokens']
    assert sum(v for k, v in results['entropy']['cost']['operator_flops'].items() if 'log_softmax' in k) > 0
    assert [a['prefix_ids'] for a in results['entropy']['allocation']] == [a['prefix_ids'] for a in results['uniform']['allocation']]
    assert max(a['prefix_length'] for a in results['entropy']['allocation']) == cfg.study.prefix_tokens


def test_weighted_gate_and_arithmetic_recover_ordinary_distribution(toy):
    cfg, engine, q = toy
    cfg.study.beta = 3
    weighted, naive, arithmetic = [], [], []
    # The long toy path consumes >53 bits; this also detects arithmetic precision collapse.
    with torch.inference_mode():
        for rep in range(200):
            r = execute(engine, q, 'entropy', 24, rep + 5000)
            weighted.append(r['distribution'].get('1', 0))
            naive.append(np.mean([s['answer'] == '1' for s in r['samples']]))
            a = execute(engine, q, 'arithmetic', 16, rep + 7000)
            arithmetic.append(a['distribution'].get('1', 0))
    assert np.mean(weighted) == pytest.approx(.375, abs=.035)
    assert np.mean(arithmetic) == pytest.approx(.375, abs=.035)
    assert np.mean(naive) > .47


def test_unknown_float_operator_fails_closed():
    with Census() as c:
        torch.erf(torch.tensor([.2]))
    with pytest.raises(RuntimeError, match='Uncovered'):
        c.require_coverage()
    with Census() as c:
        torch.mm(torch.ones(2, 3), torch.ones(3, 4))
    assert c.report()['total_flops'] == 2 * 2 * 3 * 4
    c.require_coverage()


def test_thresholds_use_observed_grid_and_best_baseline():
    # [budget, method, metric] with four methods.
    points = np.zeros((3, 4, 4))
    points[:, :, 0] = np.array([100, 200, 400])[:, None]
    points[:, :, 2] = np.array([.1, .04, .02])[:, None]
    points[:, 2, 0] *= .5
    points[:, 1, 0] *= .8
    result = speedup(points, 0, .05, 'best_baseline')
    assert result['winning_baseline'] == 'uniform'
    assert result['entropy_cost'] == 100  # observed middle point, no interpolated crossing
    assert result['speedup'] == 1.6
    assert speedup(points, 0, .001, 'best_baseline')['speedup'] is None
    assert squared_l2({'a': 1}, {'b': 1}) == 2


def test_bootstrap_preserves_paired_runs_and_ignores_leaves():
    rng = np.random.default_rng(19)
    data = rng.random((4, 3, 5, 4, 4))
    data[:, :, :, 2, :] = data[:, :, :, 1, :] * .5
    draws = bootstrap(data, 40, 333)
    assert np.allclose(draws[:, :, 2, :], draws[:, :, 1, :] * .5)

@pytest.mark.parametrize('patch', [
    {'study': {'dev_split': 'test', 'dev_offset': 1000}},
    {'prompt': {'enable_thinking': True}}, {'study': {'roots': 0}},
    {'study': {'calibration_axis': 'gpu_seconds'}}, {'sampling': {'top_k': 10}},
    {'sampling': {'temperature': .6}}, {'sampling': {'min_p': .1}},
    {'prompt': {'n_shots': 7}}, {'backend': {'batch_size': 0}},
    {'parser': {'int_tolerance': 1e-6}}, {'study': {'iid_budgets': [32, 16]}},
])
def test_protocol_rejects_invalid_rules(patch):
    with pytest.raises(ValueError):
        load_config('configs/net-demo.yaml', overrides=patch)
