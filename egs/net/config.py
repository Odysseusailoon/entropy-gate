from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml
from egs.config import (ModelCfg, BackendCfg, PromptCfg, SamplingCfg, DataCfg,
                        ParserCfg, ReferenceCfg, ChecksCfg, RunCfg, _merge)

@dataclass
class Study:
    required_gpu_name: str = 'H200'
    iid_budgets: list[int] = field(default_factory=lambda: [16, 32, 64])
    replicates: int = 10
    roots: int = 8
    prefix_tokens: int = 64
    n_min: int = 1
    beta: float = 1.0
    entropy_units: str = 'bits'
    exclude_special_positions: bool = True
    dev_split: str = 'train'
    dev_offset: int = 8  # demos use 0..7
    dev_questions: int = 10
    dev_replicates: int = 3
    candidates: list[int] = field(default_factory=lambda: [8, 16, 32, 48, 64, 96, 128])
    # Sample counts are selected using DEVELOPMENT FLOPs only; no test adaptation.
    calibration_axis: str = 'total_flops'
    calibration_max_relative_error: float = 0.35
    epsilon_js: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 902071
    queue_seed: int = 310927
    arithmetic_permute_vocab: bool = True
    reference_max_samples: int = 4096
    warmup_rollouts: int = 2
    # Declare any other question IDs used in previous gate tuning here.
    tuning_question_ids: list[str] = field(default_factory=list)

@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    backend: BackendCfg = field(default_factory=lambda: BackendCfg(batch_size=8))
    prompt: PromptCfg = field(default_factory=PromptCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    data: DataCfg = field(default_factory=lambda: DataCfg(n_questions=50, question_offset=1000))
    parser: ParserCfg = field(default_factory=ParserCfg)
    reference: ReferenceCfg = field(default_factory=lambda: ReferenceCfg(n_ref=1024,
        convergence_grid=[128, 256, 512, 1024]))
    checks: ChecksCfg = field(default_factory=lambda: ChecksCfg(truncation_threshold=0.0))
    run: RunCfg = field(default_factory=lambda: RunCfg(name='net-benefit', stage='pilot'))
    study: Study = field(default_factory=Study)
    def to_dict(self):
        return asdict(self)
    @property
    def out_path(self):
        return Path(self.run.out_dir) / self.run.name


def load_config(*paths, overrides=None):
    cfg = Config()
    for path in paths:
        with open(path, encoding='utf-8') as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(c):
    def need(ok, message):
        if not ok:
            raise ValueError(message)
    need(c.backend.name in {'hf', 'mock'}, 'backend must be hf or mock')
    need(c.backend.dtype in {'bfloat16', 'float16', 'float32'}, 'invalid dtype')
    need(c.backend.attention in {'sdpa', 'eager'}, 'use audited SDPA or eager attention')
    need(c.backend.deterministic and c.sampling.use_sampling_seed, 'deterministic replay required')
    need(c.prompt.style == 'chat_multiturn' and c.prompt.n_shots == 8 and len(set(c.prompt.shot_indices)) == 8,
         'freeze exactly eight demonstrations')
    need(not c.prompt.enable_thinking, 'this first net-benefit experiment is non-thinking')
    need(c.prompt.answer_prefix == 'Final answer:' and c.parser == ParserCfg(), 'parser policy is frozen')
    need(c.sampling.temperature == 1 and c.sampling.top_p == 1 and c.sampling.top_k == -1 and c.sampling.min_p == 0,
         'sampling must preserve the full model distribution')
    s = c.study
    need(s.required_gpu_name in {'H200', 'A100'}, 'freeze a supported homogeneous GPU model')
    for name, value in {'replicates': s.replicates, 'roots': s.roots, 'n_min': s.n_min,
            'prefix_tokens': s.prefix_tokens, 'dev_questions': s.dev_questions, 'dev_replicates': s.dev_replicates,
            'bootstrap_samples': s.bootstrap_samples, 'batch_size': c.backend.batch_size,
            'n_questions': c.data.n_questions, 'n_ref': c.reference.n_ref, 'warmup_rollouts': s.warmup_rollouts}.items():
        need(type(value) is int and value > 0, f'{name} must be a positive integer')
    need(s.roots >= 2 and s.prefix_tokens < c.sampling.max_new_tokens, 'invalid branching depth/root count')
    need(math.isfinite(s.beta) and s.beta >= 0 and s.entropy_units in {'bits', 'nats'}, 'invalid gate')
    need(s.calibration_axis == 'total_flops', 'freeze budget calibration on FLOPs; report GPU seconds separately')
    for name, grid in [('iid_budgets', s.iid_budgets), ('candidates', s.candidates)]:
        need(grid and grid == sorted(set(grid)) and all(type(n) is int and n >= s.roots * s.n_min for n in grid),
             f'{name} must be sorted, unique and cover every root minimum')
    need(set(s.iid_budgets) <= set(s.candidates), 'calibration must contain all IID budget anchors')
    need(s.epsilon_js and all(0 < x < 1 for x in s.epsilon_js), 'freeze positive JS thresholds')
    need(s.dev_split in {'train', 'test'} and c.data.split == 'test', 'invalid dev/test split')
    need(s.dev_offset >= 0 and c.data.question_offset >= 0, 'negative data offset')
    if s.dev_split == 'train':
        need(not set(range(s.dev_offset, s.dev_offset + s.dev_questions)) & set(c.prompt.shot_indices), 'development overlaps demonstrations')
    else:
        need(not set(range(s.dev_offset, s.dev_offset + s.dev_questions)) & set(range(c.data.question_offset, c.data.question_offset + c.data.n_questions)), 'development overlaps test')
    need(c.reference.convergence_grid == sorted(set(c.reference.convergence_grid)) and
         len(c.reference.convergence_grid) >= 2 and c.reference.convergence_grid[-1] == c.reference.n_ref,
         'reference convergence grid must include a comparison and end at n_ref')
    need(s.reference_max_samples >= c.reference.n_ref, 'reference maximum below starting reference')
    need(c.reference.n_ref >= 1024 or c.backend.name == 'mock', 'start real reference at 1024 per question')
    need(c.backend.name != 'mock' or c.run.stage == 'demo', 'synthetic data must be marked demo')
    need(c.checks.require_manual_review or c.backend.name == 'mock', 'real runs require the 50-rollout human audit')
    need(c.checks.manual_review_samples >= 50, 'review at least 50 rollouts')
    need(0 < c.checks.invalid_threshold <= .05 and 0 <= c.checks.truncation_threshold <= .05, 'invalid quality thresholds')
    need(0 <= c.checks.entropy_repeat_atol <= 1e-4, 'entropy replay tolerance too large')
    need(0 <= c.checks.batch_probability_tv_tolerance <= .01, 'batch probability tolerance too large')
    need(0 < s.calibration_max_relative_error < 1, 'invalid calibration tolerance')
