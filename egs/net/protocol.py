"""Freeze data separation, estimator, sampler, cache, timing and FLOP policies."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import os
import re
import yaml
from egs.data import Question, build_questions
from egs.io_utils import build_manifest, code_fingerprint, read_json, read_jsonl, sha256_file, sha256_obj, write_json, write_jsonl
from egs.protocol import role_lock, runtime_versions, protocol_sources
from .config import load_config


def target_signature(cfg, fingerprint):
    # Seeds and number of questions do not define the target; exact per-question
    # input IDs are verified separately, enabling independent reference reuse.
    return sha256_obj({'model': cfg.model.name, 'revision': cfg.model.revision,
        'tokenizer': cfg.model.tokenizer_name, 'tokenizer_revision': cfg.model.tokenizer_revision,
        'thinking': cfg.prompt.enable_thinking, 'temperature': cfg.sampling.temperature,
        'top_p': cfg.sampling.top_p, 'top_k': cfg.sampling.top_k, 'min_p': cfg.sampling.min_p,
        'max_new_tokens': cfg.sampling.max_new_tokens, 'dtype': cfg.backend.dtype,
        'attention': cfg.backend.attention, 'parser': asdict(cfg.parser),
        'eos_ids': fingerprint['eos_token_ids'], 'synthetic': cfg.backend.name == 'mock'})


def freeze(cfg, offline=False):
    root = cfg.out_path
    with role_lock(root, 'net-freeze'):
        if (root / 'manifest.json').exists():
            raise RuntimeError('run already frozen; choose a fresh directory')
        if cfg.backend.name != 'mock':
            if offline:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
            from transformers import AutoConfig
            cfg.model.tokenizer_name = cfg.model.tokenizer_name or cfg.model.name
            cfg.model.tokenizer_revision = cfg.model.tokenizer_revision or (
                cfg.model.revision if cfg.model.tokenizer_name == cfg.model.name else 'main')
            if not offline:
                from huggingface_hub import HfApi
                api = HfApi()
                cfg.model.revision = api.model_info(cfg.model.name, revision=cfg.model.revision).sha
                cfg.model.tokenizer_revision = api.model_info(cfg.model.tokenizer_name, revision=cfg.model.tokenizer_revision).sha
                cfg.data.revision = api.dataset_info(cfg.data.dataset, revision=cfg.data.revision).sha
            if any(not s or not re.fullmatch('[0-9a-f]{40}', s) for s in [cfg.model.revision, cfg.model.tokenizer_revision, cfg.data.revision]):
                raise RuntimeError('immutable SHA revisions required; offline freeze cannot resolve main')
            model_cfg = AutoConfig.from_pretrained(cfg.model.name, revision=cfg.model.revision, local_files_only=offline)
            cfg.model.n_layers, cfg.model.hidden_size = model_cfg.num_hidden_layers, model_cfg.hidden_size
            # Metadata only; the measured FLOP census never uses a 2*N estimate.
            import torch
            from transformers import AutoModelForCausalLM
            with torch.device('meta'):
                shape_model = AutoModelForCausalLM.from_config(model_cfg)
            cfg.model.params_total = sum(p.numel() for p in shape_model.parameters())
            del shape_model
        test, fp = build_questions(cfg)
        development = deepcopy(cfg)
        development.data.split = cfg.study.dev_split
        development.data.question_offset = cfg.study.dev_offset
        development.data.n_questions = cfg.study.dev_questions
        dev, dev_fp = build_questions(development)
        test_ids, dev_ids = {q.question_id for q in test}, {q.question_id for q in dev}
        if test_ids & (dev_ids | set(cfg.study.tuning_question_ids)):
            raise RuntimeError('held-out test set overlaps development or declared tuning questions')
        if {sha256_obj(q.input_ids) for q in test} & {sha256_obj(q.input_ids) for q in dev}:
            raise RuntimeError('development/test prompt duplication')
        write_jsonl(root / 'questions.jsonl', (asdict(q) for q in test))
        write_jsonl(root / 'development_questions.jsonl', (asdict(q) for q in dev))
        write_json(root / 'prompt_fingerprint.json', fp)
        write_json(root / 'development_fingerprint.json', dev_fp)
        with open(root / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)
        files = ['questions.jsonl', 'development_questions.jsonl', 'prompt_fingerprint.json', 'development_fingerprint.json', 'config.yaml']
        write_json(root / 'manifest.json', build_manifest(cfg.to_dict(), {
            'protocol': 'net-benefit-v3', 'files': {p: sha256_file(root / p) for p in files},
            'runtime_versions': runtime_versions(), 'protocol_sources': protocol_sources(),
            'target_signature': target_signature(cfg, fp), 'synthetic': cfg.backend.name == 'mock',
            'independent_unit': 'complete question/method/configuration/run; leaves are correlated',
            'cost_policy': 'complete predetermined runs; observed operator FLOPs and GPU occupied seconds',
            'reference_access': 'workers cannot read reference outcomes',
            'cache_policy': 'one paid shared prompt prefill per run; reused prefix KV; same max batch for all methods'}))
    return root


def open_run(root):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    if manifest.get('protocol') != 'net-benefit-v3':
        raise RuntimeError('use a v3 run; old method-per-GPU runs are not net-benefit experiments')
    for name, value in manifest['files'].items():
        if sha256_file(root / name) != value:
            raise RuntimeError(f'frozen file changed: {name}')
    if code_fingerprint(Path(__file__).resolve().parents[1]) != manifest['code_fingerprint']:
        raise RuntimeError('code changed after freeze')
    if runtime_versions() != manifest['runtime_versions'] or protocol_sources() != manifest['protocol_sources']:
        raise RuntimeError('dependencies or preregistration changed after freeze')
    cfg = load_config(root / 'config.yaml')
    test = [Question(**r) for r in read_jsonl(root / 'questions.jsonl')]
    dev = [Question(**r) for r in read_jsonl(root / 'development_questions.jsonl')]
    return cfg, test, dev, read_json(root / 'prompt_fingerprint.json'), manifest
