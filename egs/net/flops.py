"""Executed-operator FLOP census with explicit coverage and fail-closed unknowns.

This is an operator/shape census, not a physical hardware instruction counter.
FMA=2; add/mul/div/pow/exp/log/sin/cos/rsqrt=1. Tensor comparisons, indexing,
RNG, copies and allocation are audited as non-FLOP work and paid in GPU seconds.
No unsupported floating-point operator can silently contribute zero.
"""
from collections import Counter
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry


def tensors(obj):
    return [x for x in tree_flatten(obj)[0] if isinstance(x, torch.Tensor)]


def nfloat(obj):
    return sum(t.numel() for t in tensors(obj) if t.is_floating_point() or t.is_complex())

# These operations move/select/represent values or do integer/control work.
ZERO = set('alias detach detach_ view _unsafe_view view_as reshape expand expand_as unsqueeze squeeze '
           'transpose t permute contiguous clone copy_ _to_copy to lift_fresh lift_fresh_copy '
           'empty empty_like empty_strided zeros zeros_like ones ones_like full full_like new_empty '
           'new_zeros new_ones new_full fill_ zero_ cat stack split split_with_sizes unbind slice '
           'slice_scatter select select_scatter narrow as_strided as_strided_ index index_select '
           'index_copy_ index_put_ gather scatter_ scatter src_scatter masked_fill masked_fill_ '
           'where embedding repeat repeat_interleave repeat_interleave.Tensor numel size stride '
           'is_contiguous _local_scalar_dense item rand rand_like randn randperm random_ uniform_ '
           'normal_ arange linspace scalar_tensor tensor set_ resize_ resize_as_ record_stream '
           'eq ne lt le gt ge isinf isnan isfinite any all bitwise_and bitwise_or bitwise_not '
           'logical_and logical_or logical_not logical_xor searchsorted bucketize sort argsort '
           'argmax argmin max min maximum minimum clamp clamp_ clamp_min clamp_max '
           'nextafter tril triu tril_ triu_ _assert_async _assert_scalar _assert_tensor_metadata '
           'resolve_conj resolve_neg'.split())
ELEMENT = set('add add_ sub sub_ mul mul_ div div_ true_divide floor_divide pow square sqrt rsqrt '
              'exp exp_ exp2 expm1 log log_ log2 log10 log1p neg neg_ abs reciprocal sin cos tan '
              'floor ceil round trunc remainder fmod lerp'.split())


class Census(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts, self.calls, self.uncovered, self.nonflop = Counter(), Counter(), Counter(), Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = func._overloadpacket
        name = packet.__name__
        label = str(packet)
        self.calls[label] += 1
        inputs = tensors((args, kwargs))
        floating = [v for v in inputs if v.is_floating_point() or v.is_complex()]
        count = 0
        if not floating and not nfloat(out):
            self.nonflop[label] += 1
            return out
        if 'scaled_dot_product' in name or name in {'_flash_attention_forward', '_efficient_attention_forward'}:
            q, k, v = args[:3]
            if q.ndim != 4 or k.ndim != 4:
                self.uncovered[label] += 1
                return out
            # SDPA [B,H,L,D]. Includes QK, scale, stable softmax, and AV.
            pairs = q.shape[0] * q.shape[1] * q.shape[-2] * k.shape[-2]
            count = 2 * pairs * (q.shape[-1] + v.shape[-1]) + 5 * pairs - q.shape[0] * q.shape[1] * q.shape[-2]
        elif name in {'matmul', 'linear', 'mv', 'dot'}:
            count = 2 * nfloat(out) * args[0].shape[-1]
            if name == 'linear' and len(args) > 2 and args[2] is not None:
                count += nfloat(out)
        elif name == 'dropout':
            if args[1] == 0 or not args[2]:
                self.nonflop[label] += 1
            else:
                self.uncovered[label] += 1
        elif packet in flop_registry:
            count = int(flop_registry[packet](*args, **kwargs, out_val=out))
            if name in {'addmm', 'baddbmm'}:
                count += nfloat(out)  # bias addition
        elif name in ELEMENT:
            count = nfloat(out) * (2 if name == 'lerp' else 1)
            if name in {'add', 'add_', 'sub', 'sub_'} and kwargs.get('alpha', 1) != 1:
                count += nfloat(out)
        elif name in {'sum', 'nansum', 'cumsum', 'mean'}:
            src = args[0]
            if src.is_floating_point():
                count = src.numel() if name == 'cumsum' else max(0, src.numel() - nfloat(out))
                if name == 'mean':
                    count += nfloat(out)
        elif name in {'scatter_add', 'scatter_add_', 'index_add', 'index_add_'}:
            count = nfloat(args[-1])
        elif name == 'isclose':
            count = 5 * max(t.numel() for t in floating)
        elif name in {'_softmax', 'softmax', '_safe_softmax', '_log_softmax', 'log_softmax'}:
            dim = args[1] if len(args) > 1 else kwargs.get('dim', -1)
            src = args[0]
            count = 4 * src.numel() - (0 if 'log' in name else src.numel() // src.shape[dim])
        elif name in {'silu', 'sigmoid', 'tanh', 'gelu'}:
            count = nfloat(out) * {'silu': 4, 'sigmoid': 4, 'tanh': 6, 'gelu': 8}[name]
        elif name in {'native_layer_norm', 'layer_norm', 'rms_norm'}:
            count = args[0].numel() * (8 if 'layer' in name else 5)
        elif name in ZERO:
            self.nonflop[label] += 1
        else:
            self.uncovered[label] += 1
        self.counts[label] += int(count)
        return out

    def report(self):
        return {'total_flops': int(sum(self.counts.values())), 'operator_flops': dict(self.counts),
                'operator_calls': dict(self.calls), 'nonflop_calls': dict(self.nonflop),
                'uncovered_float_operators': dict(self.uncovered), 'coverage_passed': not self.uncovered,
                'flop_definition': 'executed-operator census v1; FMA=2, scalar transcendental=1; not hardware instructions'}

    def require_coverage(self):
        if self.uncovered:
            raise RuntimeError(f'Uncovered floating-point operators: {dict(self.uncovered)}; add audited rules before scoring')
        if sum(self.counts.values()) <= 0:
            raise RuntimeError('empty FLOP census')
