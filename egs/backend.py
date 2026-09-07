"""Deterministic model loading and domain-separated seeds for the shared engine."""
from __future__ import annotations
import os
from .io_utils import sha256_obj


def seed_for(base, *parts):
    # A 63-bit seed avoids 32-bit birthday collisions in the million-rollout reference.
    return int(sha256_obj([base, *parts])[:16], 16) & ((1 << 63) - 1)


class HFBackend:
    def __init__(self, cfg, eos_ids, special_ids, model=None, tokenizer=None):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        self.torch, self.cfg = torch, cfg
        self.eos_ids, self.special_ids = set(eos_ids), set(special_ids)
        if not self.eos_ids:
            raise ValueError("EOS policy must be frozen")
        self.device = cfg.backend.device
        if cfg.backend.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if model is None:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name, revision=cfg.model.revision,
                torch_dtype=getattr(torch, cfg.backend.dtype),
                attn_implementation=cfg.backend.attention, trust_remote_code=False)
        self.model = model.to(self.device).eval()
        if tokenizer is None:
            from .data import get_tokenizer
            tokenizer = get_tokenizer(cfg.model.tokenizer_name or cfg.model.name, cfg.model.tokenizer_revision)
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(self.eos_ids)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
