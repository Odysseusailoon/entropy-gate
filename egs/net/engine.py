"""One cache/batch framework for IID, both forks, and Arithmetic Sampling.

Prompt prefill is performed ONCE inside every independent run and paid by every
method. Prefix K/V tensors are reused by value, with fresh batched cache storage
for writable continuations; no prefix token is forwarded again after a fork.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from egs.backend import HFBackend
from .arithmetic import Sampler

@dataclass
class State:
    ids: list[int]
    cache: tuple | None
    logits: torch.Tensor | None
    reason: str = 'active'
    entropies: list = field(default_factory=list)


class Engine:
    def __init__(self, cfg, fingerprint, model=None, tokenizer=None):
        self.cfg = cfg
        self.synthetic = cfg.backend.name == 'mock'
        self.device = 'cpu' if self.synthetic else cfg.backend.device
        self.eos_ids = set(fingerprint['eos_token_ids'])
        self.special_ids = set(fingerprint['special_token_ids'])
        self.base = None if self.synthetic else HFBackend(cfg, self.eos_ids, self.special_ids, model=model, tokenizer=tokenizer)
        self.reset_ledger()

    def reset_ledger(self):
        self.ledger = {'prefill_tokens': 0, 'decode_forward_tokens': 0, 'generated_tokens': 0,
                       'prompt_prefills': 0, 'forward_calls': 0, 'cache_forks': 0,
                       'batch_policy': 'fixed max batch; stable order; remove finished rows; no cross-run cache'}

    @property
    def is_cuda(self):
        return str(self.device).startswith('cuda')

    def synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def decode(self, ids):
        if not self.synthetic:
            return self.base.decode(ids)
        answer = next((str(i - 100) for i in reversed(ids) if i in (100, 101)), None)
        return 'Synthetic reasoning.\n' + (f'Final answer: {answer}' if answer is not None else '')

    def _toy_logits(self, paths):
        # Exact finite process: root 10 with p=.25, then answer1=.9; root11
        # with p=.75, then answer1=.2. P(answer1)=.375, EOS at position 67.
        logits = torch.full((len(paths), 102), -torch.inf, device=self.device)
        for row, path in enumerate(paths):
            n = len(path)
            if n == 0:
                values = {10: .25, 11: .75}
            elif n < 65:
                values = {20: .5, 21: .5} if path[0] == 10 else {20: 1.}
            elif n == 65:
                p = .9 if path[0] == 10 else .2
                values = {101: p, 100: 1 - p}
            else:
                values = {0: 1.}
            indices = torch.tensor(list(values), dtype=torch.long, device=self.device)
            logits[row, indices] = torch.log(torch.tensor(list(values.values()), device=self.device))
        return logits

    def prefill(self, question):
        self.ledger['prompt_prefills'] += 1
        self.ledger['prefill_tokens'] += len(question.input_ids)
        self.ledger['forward_calls'] += 1
        if self.synthetic:
            return State([], (), self._toy_logits([[]])[0:1])
        if len(question.input_ids) + self.cfg.sampling.max_new_tokens > self.base.model.config.max_position_embeddings:
            raise RuntimeError('context overflow; automatic prompt truncation is forbidden')
        tokens = torch.tensor([question.input_ids], dtype=torch.long, device=self.device)
        out = self.base.model(input_ids=tokens, use_cache=True, logits_to_keep=1)
        return State([], out.past_key_values.to_legacy_cache(), out.logits[:, -1, :])

    def fork(self, parent, n):
        self.ledger['cache_forks'] += n
        # States share immutable snapshots. _forward always builds a NEW DynamicCache.
        return [State(list(parent.ids), parent.cache, parent.logits, parent.reason, list(parent.entropies)) for _ in range(n)]

    def _forward(self, states, tokens):
        self.ledger['forward_calls'] += 1
        self.ledger['decode_forward_tokens'] += len(states)
        if self.synthetic:
            logits = self._toy_logits([s.ids for s in states])
            for i, state in enumerate(states):
                state.logits, state.cache = logits[i:i + 1], ()
            return
        from transformers.cache_utils import DynamicCache
        layers = len(states[0].cache)
        cache = DynamicCache.from_legacy_cache(tuple((
            torch.cat([s.cache[l][0] for s in states], dim=0),
            torch.cat([s.cache[l][1] for s in states], dim=0)) for l in range(layers)))
        out = self.base.model(input_ids=tokens[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        kv = out.past_key_values.to_legacy_cache()
        for i, state in enumerate(states):
            state.logits = out.logits[i:i + 1, -1, :]
            state.cache = tuple((k[i:i + 1], v[i:i + 1]) for k, v in kv)

    def advance(self, states, sampler, stop_at=None, record_entropy=False):
        """Finish ALL predetermined paths; stop_at is a fixed PREFIX boundary only."""
        cap = self.cfg.sampling.max_new_tokens
        for start in range(0, len(states), self.cfg.backend.batch_size):
            active = [(i, states[i]) for i in range(start, min(start + self.cfg.backend.batch_size, len(states)))
                      if states[i].reason == 'active']
            while active:
                positions = {len(s.ids) for _, s in active}
                if len(positions) != 1:
                    raise RuntimeError('batch must have equal generated-prefix lengths')
                position = next(iter(positions))
                logits = torch.cat([s.logits for _, s in active], dim=0)
                h = None
                if record_entropy:
                    lp = torch.log_softmax(logits.float(), -1)
                    terms = torch.where(torch.isfinite(lp), lp.exp() * lp, torch.zeros_like(lp))
                    h = -terms.sum(-1)
                    if self.cfg.study.entropy_units == 'bits':
                        h = h / torch.log(torch.tensor(2., device=self.device))
                drawn = sampler.draw(logits, [i for i, _ in active], position)
                ids = drawn.tolist()
                continuing, next_tokens = [], []
                for row, ((index, state), token) in enumerate(zip(active, ids)):
                    state.ids.append(token)
                    self.ledger['generated_tokens'] += 1
                    if record_entropy:
                        state.entropies.append(None if (self.cfg.study.exclude_special_positions and token in self.special_ids) else h[row])
                    if token in self.eos_ids:
                        state.reason, state.cache, state.logits = 'eos', None, None
                    elif len(state.ids) >= cap:
                        state.reason, state.cache, state.logits = 'length', None, None
                    else:
                        continuing.append((index, state))
                        next_tokens.append(drawn[row])
                # Consume the last prefix token before forking: its cached K/V and
                # next-token logits are then computed only once for all children.
                if continuing:
                    self._forward([s for _, s in continuing], torch.stack(next_tokens))
                if stop_at is not None and position + 1 >= stop_at:
                    break
                active = continuing
        return states
