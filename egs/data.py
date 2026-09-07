"""Materialize the exact questions, demonstrations, template and token-ID prompts."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from .answer_parser import clean_gsm8k_solution, parse_gold
from .io_utils import sha256_obj, sha256_str

@dataclass
class Question:
    question_id: str
    index: int
    question: str
    gold: str
    input_ids: list[int]
    prompt_text: str
    @property
    def n_prompt_tokens(self):
        return len(self.input_ids)

@lru_cache(maxsize=4)
def get_tokenizer(name, revision):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)


def render_prompt(cfg, tok, shots, question):
    messages = [{"role": "system", "content": cfg.prompt.system_prompt}]
    for q, answer in shots:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": question.strip()})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=cfg.prompt.enable_thinking)
    return text, tok(text, add_special_tokens=False)["input_ids"]


def build_questions(cfg):
    if cfg.backend.name == "mock":
        questions = [Question(f"toy-{i}", i, "Synthetic Bernoulli process", "0", [900, i, -1], "SYNTHETIC")
                     for i in range(cfg.data.question_offset, cfg.data.question_offset + cfg.data.n_questions)]
        return questions, {"synthetic": True, "eos_token_ids": [0], "special_token_ids": [0],
                           "question_ids_hash": sha256_obj([q.question_id for q in questions]),
                           "all_prompt_ids_hash": sha256_obj([q.input_ids for q in questions])}
    from datasets import load_dataset
    from transformers import GenerationConfig
    tok = get_tokenizer(cfg.model.tokenizer_name, cfg.model.tokenizer_revision)
    if "enable_thinking" not in tok.chat_template:
        raise ValueError("chat template does not expose the frozen thinking switch")
    ds = load_dataset(cfg.data.dataset, cfg.data.subset, revision=cfg.data.revision)
    # Datasets' offline fallback may select its latest cached configuration even
    # when revision=... was supplied. Reject a different cache commit explicitly.
    cache_paths = [entry['filename'] for split in ds.values() for entry in split.cache_files]
    if os.environ.get('HF_DATASETS_OFFLINE') == '1' and (
            not cache_paths or any(cfg.data.revision not in Path(path).parts for path in cache_paths)):
        raise RuntimeError('offline dataset cache does not match the pinned revision')
    shots = [(ds["train"][i]["question"].strip(), clean_gsm8k_solution(ds["train"][i]["answer"]))
             for i in cfg.prompt.shot_indices]
    test = ds[cfg.data.split]
    idx = list(range(len(test)))
    if cfg.data.shuffle_seed is not None:
        import random
        random.Random(cfg.data.shuffle_seed).shuffle(idx)
    lo = cfg.data.question_offset
    idx = idx[lo:lo + cfg.data.n_questions]
    if len(idx) != cfg.data.n_questions:
        raise ValueError("requested question subset exceeds the split")
    qs = []
    for i in idx:
        rec = test[i]
        text, ids = render_prompt(cfg, tok, shots, rec["question"])
        qs.append(Question(f"{cfg.data.split}-{i}", i, rec["question"].strip(), parse_gold(rec["answer"]), ids, text))
    generation = GenerationConfig.from_pretrained(cfg.model.name, revision=cfg.model.revision)
    eos = generation.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(v is None for v in eos):
        raise ValueError("model generation config must define EOS")
    return qs, {"synthetic": False, "eos_token_ids": sorted(set(eos)), "special_token_ids": sorted(tok.all_special_ids),
                "shots": shots, "shots_hash": sha256_obj(shots), "chat_template": tok.chat_template,
                "chat_template_hash": sha256_str(tok.chat_template), "system_prompt_hash": sha256_str(cfg.prompt.system_prompt),
                "all_prompt_ids_hash": sha256_obj([q.input_ids for q in qs]),
                "question_ids_hash": sha256_obj([q.question_id for q in qs]),
                "prompt_hashes": {q.question_id: sha256_obj(q.input_ids) for q in qs},
                "enable_thinking": cfg.prompt.enable_thinking}
