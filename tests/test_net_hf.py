"""Actual tiny Qwen3 checks for physical prefix reuse, cache isolation and FLOPs."""
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from egs.net.config import load_config
from egs.net.engine import Engine
from egs.net.arithmetic import Sampler
from egs.net.experiment import measure
from egs.data import Question

class Tok:
    pad_token_id = 0
    def decode(self, ids, **kwargs):
        return 'Final answer: 1'

@pytest.fixture(params=['eager', 'sdpa'])
def tiny(request):
    torch.set_num_threads(1); torch.manual_seed(5)
    cfg = load_config(overrides={'backend': {'device': 'cpu', 'dtype': 'float32', 'attention': request.param},
        'sampling': {'max_new_tokens': 12}, 'study': {'prefix_tokens': 4, 'roots': 4,
        'iid_budgets': [8, 16], 'candidates': [8, 16]}})
    mc = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        eos_token_id=31, pad_token_id=0)
    mc._attn_implementation = request.param
    engine = Engine(cfg, {'eos_token_ids': [31], 'special_token_ids': [0, 31]}, Qwen3ForCausalLM(mc), Tok())
    return cfg, engine, Question('q', 0, 'x', '1', [1, 2, 3], 'x')


def test_cached_prefix_equals_full_model_and_parent_not_mutated(tiny):
    cfg, engine, q = tiny
    with torch.inference_mode():
        prompt = engine.prefill(q)
        roots = engine.fork(prompt, 4)
        engine.advance(roots, Sampler('iid', 4, 555, 'cpu'), stop_at=4, record_entropy=True)
        active = next(r for r in roots if r.reason == 'active')
        full = torch.tensor([q.input_ids + active.ids])
        expected = engine.base.model(input_ids=full, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        assert torch.allclose(active.logits, expected, atol=1e-6, rtol=1e-5)
        snapshots = [(k.clone(), v.clone()) for k, v in active.cache]
        children = engine.fork(active, 8)
        engine.advance(children, Sampler('iid', 8, 999, 'cpu'))
        for (before_k, before_v), (after_k, after_v) in zip(snapshots, active.cache):
            assert torch.equal(before_k, after_k) and torch.equal(before_v, after_v)
        assert engine.ledger['prompt_prefills'] == 1
        assert engine.ledger['prefill_tokens'] == 3


def test_complete_operator_census_for_real_model(tiny):
    cfg, engine, q = tiny
    for method in ['iid', 'uniform', 'entropy', 'arithmetic']:
        result = measure(engine, q, method, 8, 91)
        assert result['cost']['total_flops'] > 0
        assert not result['cost']['uncovered_float_operators']
        assert result['cost']['instrumentation_replay_verified']
        assert result['ledger']['prefill_tokens'] == 3


def test_batch_probability_probe_detects_real_distribution_shift(tiny, monkeypatch):
    from egs.net.preflight import batch_numeric_probe
    cfg, engine, q = tiny
    result = batch_numeric_probe(engine, q, [1, 2, 3, 4, 5])
    assert result['passed']
    assert max(p['max_row_total_variation'] for p in result['probes']) < 1e-5
    original = engine._forward
    def shifted(states, tokens):
        original(states, tokens)
        if len(states) > 1:
            for state in states:
                state.logits = state.logits.clone()
                state.logits[:, 0] += 10
    monkeypatch.setattr(engine, '_forward', shifted)
    assert not batch_numeric_probe(engine, q, [1, 2, 3, 4, 5])['passed']
