"""Shared configuration records and strict mapping merge for the v3 study."""
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class ModelCfg:
    name: str = "Qwen/Qwen3-8B"
    revision: str = "main"
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    params_total: float = 8.19e9
    n_layers: int = 32
    hidden_size: int = 4096


@dataclass
class BackendCfg:
    name: str = "hf"  # hf: exact full-vocabulary entropy; mock: offline tests only
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    batch_size: int = 8
    attention: str = "sdpa"
    deterministic: bool = True


@dataclass
class PromptCfg:
    style: str = "chat_multiturn"
    n_shots: int = 8
    shot_indices: list[int] = field(default_factory=lambda: list(range(8)))
    system_prompt: str = (
        "You are a careful assistant that solves grade school math word problems. "
        "Reason step by step, then state the final numeric answer on its own final "
        "line in exactly this format:\nFinal answer: <number>"
    )
    enable_thinking: bool = False
    answer_prefix: str = "Final answer:"


@dataclass
class SamplingCfg:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_new_tokens: int = 1024  # emergency guard; length stops are censored, never EOS
    use_sampling_seed: bool = True
    seed_base: int = 20260906


@dataclass
class DataCfg:
    dataset: str = "openai/gsm8k"
    subset: str = "main"
    revision: str = "main"
    split: str = "test"
    n_questions: int = 500
    question_offset: int = 0
    shuffle_seed: int | None = None


@dataclass
class ReferenceCfg:
    n_ref: int = 2048
    convergence_grid: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048])
    convergence_js_threshold: float = 0.01


@dataclass
class ParserCfg:
    scan_reasoning_for_numbers: bool = False
    strip_commas: bool = True
    strip_currency: bool = True
    int_tolerance: float = 0.0
    invalid_token: str = "__INVALID__"


@dataclass
class ChecksCfg:
    invalid_threshold: float = 0.05
    truncation_threshold: float = 0.05
    manual_review_samples: int = 50
    require_manual_review: bool = True
    entropy_repeat_atol: float = 1e-5
    batch_probability_tv_tolerance: float = 0.01


@dataclass
class RunCfg:
    name: str = "formal"
    out_dir: str = "runs"
    resume: bool = True
    stage: str = "formal"


def _merge(dc: Any, patch: dict):
    if not isinstance(patch, dict):
        raise ValueError("config sections must be mappings")
    known = {f.name for f in fields(dc)}
    for k, value in patch.items():
        if k not in known:
            raise KeyError(f"unknown config key {k!r} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if is_dataclass(cur):
            _merge(cur, value)
        else:
            setattr(dc, k, value)
    return dc
