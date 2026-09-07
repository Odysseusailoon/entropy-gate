"""Vilnis et al., ICML 2023, Algorithm 1 and Definition 3.

Codes i/(N+1)+U mod 1, i=1..N, ONE shared uniform shift per independent run.
Each selected CDF interval is mapped affinely back to [0,1). A common optional
per-position vocabulary permutation follows the authors' T5X implementation.
No top-k truncation, no deduplication, no fresh code at each token.
"""
from __future__ import annotations
import torch
from egs.backend import seed_for


def lattice_codes(n, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    shift = torch.rand((), generator=generator, dtype=torch.float64, device=device)
    return torch.remainder(torch.arange(1, n + 1, dtype=torch.float64, device=device) / (n + 1) + shift, 1.)


def inverse_cdf(probs, codes):
    """Full support, double-precision accumulation. Returns token and residual code."""
    probs = probs.to(torch.float64)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    cdf = probs.cumsum(-1)
    cdf[:, -1] = 1.
    if not bool(((codes >= 0) & (codes < 1)).all()):
        raise RuntimeError('arithmetic code outside [0,1)')
    tokens = torch.searchsorted(cdf.contiguous(), codes[:, None].contiguous(), right=True).squeeze(-1)
    upper = cdf.gather(1, tokens[:, None]).squeeze(1)
    lower = torch.cat((torch.zeros_like(cdf[:, :1]), cdf[:, :-1]), dim=1).gather(1, tokens[:, None]).squeeze(1)
    width = upper - lower
    if not bool((width > 0).all()):
        raise RuntimeError('selected zero-mass arithmetic interval')
    residual = (codes - lower) / width
    # A boundary can round up by one ULP. Do not regenerate randomness or reset codes.
    one = torch.ones((), dtype=torch.float64, device=probs.device)
    residual = torch.minimum(residual, torch.nextafter(one, torch.zeros_like(one)))
    return tokens, residual


class Sampler:
    def __init__(self, method, n, seed, device, permute=True, path_seeds=None, max_tokens=1024):
        self.method, self.seed, self.device, self.permute = method, seed, device, permute
        self.codes = None
        self.n, self.max_tokens = n, max_tokens
        seeds = path_seeds or [seed_for(seed, 'path', i) for i in range(n)]
        self.generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

    def draw(self, logits, indices, position):
        # Exactly the same softmax and inverse-CDF primitives for all four methods.
        probs = torch.softmax(logits.float(), -1)
        ix = torch.tensor(indices, dtype=torch.long, device=self.device)
        if self.method == 'arithmetic':
            if self.codes is None:
                self.codes = ExactCodes(self.n, self.seed, self.max_tokens, probs.shape[-1])
            codes = self.codes.floats(indices, self.device)
        else:
            codes = torch.stack([torch.rand((), generator=self.generators[i], dtype=torch.float64,
                                           device=self.device) for i in indices])
        permutation = None
        if self.method == 'arithmetic' and self.permute:
            g = torch.Generator(device=self.device).manual_seed(seed_for(self.seed, 'vocabulary-order', position))
            permutation = torch.randperm(probs.shape[-1], generator=g, device=self.device)
            probs = probs.index_select(-1, permutation)
        probs = probs.to(torch.float64)
        probs = probs / probs.sum(-1, keepdim=True)
        cdf = probs.cumsum(-1)
        cdf[:, -1] = 1.
        tokens = torch.searchsorted(cdf.contiguous(), codes[:, None].contiguous(), right=True).squeeze(-1)
        if self.method == 'arithmetic':
            tokens = self.codes.update(cdf, tokens, indices)
        if permutation is not None:
            tokens = permutation[tokens]
        return tokens

class ExactCodes:
    """Prevent long-sequence code collapse from repeated float64 rescaling.

    One shared B-bit random shift, B=(L+1)*ceil(log2(V))+128. The finite-code
    discretization TV bound over <=V^(L+1) trajectories is at most 2^-128.
    Residual updates use exact rational arithmetic on the frozen float64 CDF
    intervals. Python bigint/copy/selection costs are paid in end-to-end seconds.
    """
    def __init__(self, n, seed, max_tokens, vocab_size):
        import random
        import math
        from fractions import Fraction
        self.Fraction = Fraction
        self.bits = (max_tokens + 1) * math.ceil(math.log2(max(2, vocab_size))) + 128
        shift = Fraction(random.Random(seed).getrandbits(self.bits), 1 << self.bits)
        self.values = []
        for i in range(1, n + 1):
            value = Fraction(i, n + 1) + shift
            self.values.append(value - int(value))
        self.corrections = 0

    def floats(self, indices, device):
        import math
        return torch.tensor([min(float(self.values[i]), math.nextafter(1., 0.)) for i in indices],
                            dtype=torch.float64, device=device)

    def update(self, cdf, tokens, indices):
        import bisect
        ix = tokens[:, None]
        highs = cdf.gather(1, ix).squeeze(1).tolist()
        lows = torch.cat((torch.zeros_like(cdf[:, :1]), cdf[:, :-1]), 1).gather(1, ix).squeeze(1).tolist()
        selected = tokens.tolist()
        for row, (index, lower, upper) in enumerate(zip(indices, lows, highs)):
            code = self.values[index]
            lo, hi = self.Fraction(lower), self.Fraction(upper)
            if not lo <= code < hi:
                # Search against the exact code, not its rounded float proxy.
                values = cdf[row].tolist()
                chosen = bisect.bisect_right(values, code)
                if chosen >= len(values):
                    raise RuntimeError('exact arithmetic code outside CDF')
                selected[row] = chosen
                lo = self.Fraction(values[chosen - 1]) if chosen else self.Fraction(0)
                hi = self.Fraction(values[chosen])
                self.corrections += 1
            self.values[index] = (code - lo) / (hi - lo)
        return torch.tensor(selected, dtype=torch.long, device=cdf.device)
