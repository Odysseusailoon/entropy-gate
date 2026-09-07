"""Observed cost/error frontiers, threshold speedups and paired hierarchical CIs.

No common-cost interpolation, no monotonic smoothing, no leaf bootstrap.
Primary net costs include all development calibration and worker warmup costs
amortized over the preregistered number of complete test runs per method.
"""
from pathlib import Path
import numpy as np
from egs.io_utils import read_json, read_jsonl, write_json
from egs.metrics import js_divergence, INVALID, TRUNCATED
from egs.protocol import protocol_id, require_ready
from egs.io_utils import write_csv
from . import METHODS, BASELINES
from .pipeline import load_result, read_plan, result_path, warmup_costs
from .queue import Queue
from .reference import seal_reference


def squared_l2(p, q):
    return sum((p.get(y, 0) - q.get(y, 0)) ** 2 for y in p.keys() | q.keys())


def aggregate(data):
    # [question, budget, independent run, method, (flops, gpu_seconds, JS, L2)]
    out = data.mean(axis=(0, 2))
    out[..., :2] *= data.shape[0]  # total cohort cost; errors remain macro averages
    return out


def threshold_cost(points, method, axis, epsilon):
    values = points[:, method, :]
    valid = values[(values[:, 2] <= epsilon) & np.isfinite(values[:, axis])]
    return float(valid[:, axis].min()) if len(valid) else None


def speedup(points, axis, epsilon, baseline):
    entropy_cost = threshold_cost(points, METHODS.index('entropy'), axis, epsilon)
    contenders = BASELINES if baseline == 'best_baseline' else [baseline]
    costs = {m: threshold_cost(points, METHODS.index(m), axis, epsilon) for m in contenders}
    finite = {m: c for m, c in costs.items() if c is not None}
    if entropy_cost is None or not finite:
        return {'speedup': None, 'entropy_cost': entropy_cost, 'baseline_cost': min(finite.values()) if finite else None,
                'winning_baseline': min(finite, key=finite.get) if finite else None,
                'status': 'entropy_threshold_not_reached' if entropy_cost is None else 'baseline_threshold_not_reached'}
    winner = min(finite, key=finite.get)
    return {'speedup': costs[winner] / entropy_cost, 'entropy_cost': entropy_cost,
            'baseline_cost': costs[winner], 'winning_baseline': winner, 'status': 'observed_grid'}


def bootstrap(data, count, seed):
    rng = np.random.default_rng(seed)
    q, b, r = data.shape[:3]
    draws = []
    for _ in range(count):
        questions = rng.integers(q, size=q)
        # For each selected question+configuration, resample WHOLE matched runs.
        # The same indices apply to all four methods and both cost metrics.
        repeats = rng.integers(r, size=(q, b, r))
        chosen = np.take_along_axis(data[questions], repeats[..., None, None], axis=2)
        draws.append(aggregate(chosen))
    return np.asarray(draws)


def analyze(root, cfg, questions, plots=True):
    root = Path(root)
    require_ready(root, cfg)
    queue = Queue(root)
    queue.require_complete('test')
    pending_extensions = queue.db.execute("SELECT phase,state,count(*) FROM tasks WHERE phase LIKE 'reference-%' GROUP BY phase,state").fetchall()
    if any(state != 'done' for _, state, _ in pending_extensions):
        raise RuntimeError('reference extension is still running or failed')
    queue.close()
    plan = read_plan(root)
    seal_reference(root, cfg, questions)
    ref = {r['question_id']: r['distribution'] for r in read_jsonl(root / 'reference_distributions.jsonl')}
    qn, bn, rn, mn = len(questions), len(cfg.study.iid_budgets), cfg.study.replicates, len(METHODS)
    data = np.empty((qn, bn, rn, mn, 4))
    warmups = warmup_costs(root, {'test'})
    raw, instrumentation = [], {'profiling_replay_seconds': 0., 'total_measurement_wall_seconds': 0.}
    quality = {m: {'invalid': [], 'truncated': []} for m in METHODS}
    for qi, q in enumerate(questions):
        for bi, b in enumerate(cfg.study.iid_budgets):
            for ri in range(rn):
                matched_workers = set()
                for mi, method in enumerate(METHODS):
                    row = load_result(result_path(root, 'test', q.question_id, b, ri, method))
                    if row['protocol_id'] != protocol_id(root) or not row['cost']['coverage_passed']:
                        raise RuntimeError('invalid test result or incomplete FLOP coverage')
                    matched_workers.add((row['worker'], row['hardware'].get('visible_devices')))
                    p = row['distribution']
                    error = js_divergence(p, ref[q.question_id])
                    l2 = squared_l2(p, ref[q.question_id])
                    setup = plan['setup_share_per_test_run'][method]
                    n_jobs = plan['jobs_per_method']
                    fp_add = setup['total_flops'] + warmups[method]['total_flops'] / n_jobs
                    flops = row['cost']['total_flops'] + fp_add
                    gpu = row['cost']['gpu_seconds']
                    if gpu is not None and setup['gpu_seconds'] is not None and warmups[method]['gpu_seconds'] is not None:
                        gpu += setup['gpu_seconds'] + warmups[method]['gpu_seconds'] / n_jobs
                    else:
                        gpu = float('nan')
                    data[qi, bi, ri, mi] = [flops, gpu, error, l2]
                    quality[method]['invalid'].append(p.get(INVALID, 0))
                    quality[method]['truncated'].append(p.get(TRUNCATED, 0))
                    raw.append({'question_id': q.question_id, 'budget_iid_equivalent': b, 'replicate': ri,
                        'method': method, 'samples': row['samples_requested'], 'total_flops': flops,
                        'gpu_seconds': gpu if np.isfinite(gpu) else None, 'js_bits': error, 'squared_l2': l2,
                        'inference_flops': row['cost']['total_flops'], 'inference_gpu_seconds': row['cost']['gpu_seconds'],
                        'amortized_setup_flops': fp_add, 'invalid_rate': p.get(INVALID, 0),
                        'truncation_rate': p.get(TRUNCATED, 0), 'worker': row['worker'],
                        'synthetic': cfg.backend.name == 'mock'})
                    for key in instrumentation:
                        instrumentation[key] += row['cost'][key]
                if len(matched_workers) != 1:
                    raise RuntimeError('paired methods did not run on the same worker/GPU')
    for method, qs in quality.items():
        if np.mean(qs['invalid']) >= cfg.checks.invalid_threshold or np.mean(qs['truncated']) > cfg.checks.truncation_threshold:
            raise RuntimeError(f'stop-the-line: {method} invalid/truncation rate exceeded the frozen threshold')
    out = root / 'net_analysis'
    write_csv(out / 'per_run_results.csv', raw)
    points = aggregate(data)
    draws = bootstrap(data, cfg.study.bootstrap_samples, cfg.study.bootstrap_seed)
    curve_rows = []
    for bi, b in enumerate(cfg.study.iid_budgets):
        for mi, method in enumerate(METHODS):
            row = {'method': method, 'iid_equivalent_budget': b, 'samples': plan['sample_counts'][method][str(b)],
                   'synthetic': cfg.backend.name == 'mock', 'n_questions': qn, 'independent_runs': rn}
            for k, name in enumerate(['total_flops', 'gpu_seconds', 'js_bits', 'squared_l2']):
                if np.isfinite(points[bi, mi, k]):
                    lo, hi = np.quantile(draws[:, bi, mi, k], [.025, .975])
                    row.update({name: float(points[bi, mi, k]), f'{name}_ci_low': float(lo), f'{name}_ci_high': float(hi)})
                else:
                    row.update({name: None, f'{name}_ci_low': None, f'{name}_ci_high': None})
            curve_rows.append(row)
    write_csv(out / 'observed_cost_error_curves.csv', curve_rows)
    speedups = []
    for axis, metric in enumerate(['total_flops', 'gpu_seconds']):
        for eps in cfg.study.epsilon_js:
            for baseline in (*BASELINES, 'best_baseline'):
                if not np.isfinite(points[..., axis]).all():
                    result = {'speedup': None, 'status': 'cost_not_measured'}
                    ratios = []
                else:
                    result = speedup(points, axis, eps, baseline)
                    ratios = [speedup(d, axis, eps, baseline)['speedup'] for d in draws]
                valid = [v for v in ratios if v is not None and np.isfinite(v)]
                fraction = len(valid) / cfg.study.bootstrap_samples
                # Never silently drop threshold failures then advertise a narrow CI.
                lo, hi = (np.quantile(valid, [.025, .975]) if fraction >= .95 and result['speedup'] is not None else (None, None))
                speedups.append({'cost_metric': metric, 'epsilon_js_bits': eps, 'baseline': baseline, **result,
                    'ci_low': float(lo) if lo is not None else None, 'ci_high': float(hi) if hi is not None else None,
                    'bootstrap_reached_fraction': fraction, 'ci_status': 'paired_hierarchical' if lo is not None else 'threshold_range_insufficient',
                    'estimator': 'minimum observed cost meeting epsilon; no interpolation or extrapolation'})
    write_csv(out / 'speedups.csv', speedups)
    paired = []
    for bi, b in enumerate(cfg.study.iid_budgets):
        ei = METHODS.index('entropy')
        for control in BASELINES:
            ci = METHODS.index(control)
            for ki, metric in enumerate(['total_flops', 'gpu_seconds', 'js_bits', 'squared_l2']):
                values = draws[:, bi, ei, ki] - draws[:, bi, ci, ki]
                if np.isfinite(values).all():
                    lo, hi = np.quantile(values, [.025, .975])
                    paired.append({'iid_equivalent_budget': b, 'control': control, 'metric': metric,
                        'entropy_minus_control': float(points[bi, ei, ki] - points[bi, ci, ki]),
                        'ci_low': float(lo), 'ci_high': float(hi),
                        'comparison': 'paired pre-calibrated configurations; report observed cost differences'})
    write_csv(out / 'paired_comparisons.csv', paired)
    decisions = []
    for eps in cfg.study.epsilon_js:
        subset = [r for r in speedups if r['epsilon_js_bits'] == eps]
        def wins(axis, baseline):
            r = next(r for r in subset if r['cost_metric'] == axis and r['baseline'] == baseline)
            return r['ci_low'] is not None and r['ci_low'] > 1
        measurable = all(r['speedup'] is not None and r['ci_low'] is not None for r in subset
                         if r['baseline'] in {'iid', 'uniform', 'best_baseline'})
        both = all(wins(axis, m) for axis in ['total_flops', 'gpu_seconds'] for m in ['iid', 'uniform'])
        if not measurable:
            decision = 'inconclusive: insufficient threshold coverage or GPU measurements; do not expand automatically'
        elif both:
            decision = 'net benefit over iid and uniform; expansion is supported at this threshold'
        elif all(wins('total_flops', m) for m in ['iid', 'uniform']):
            decision = 'FLOP gain without demonstrated GPU-time gain; optimize gate/cache/scheduling'
        elif all(wins(axis, 'iid') for axis in ['total_flops', 'gpu_seconds']) and not all(wins(axis, 'uniform') for axis in ['total_flops', 'gpu_seconds']):
            decision = 'gain over iid but entropy contribution over uniform is unproven'
        else:
            decision = 'no demonstrated end-to-end net benefit; do not expand on this evidence'
        decisions.append({'epsilon_js_bits': eps, 'decision': decision,
                          'beats_best_baseline_both_costs': all(wins(axis, 'best_baseline') for axis in ['total_flops', 'gpu_seconds'])})
    write_json(out / 'decision.json', {'results': decisions, 'primary_comparison': 'entropy vs uniform',
        'costs_include': 'prompt/cache/continuation/gate/host scheduling plus amortized development and warmups',
        'instrumentation_overhead': instrumentation, 'reference_uncertainty': 'conditional on accepted independent reference',
        'bootstrap_unit': 'question then complete paired run within question/configuration; never leaves',
        'cost_definition': 'executed operator FLOP census; GPU occupied synchronized wall time',
        'threshold_rule': 'observed discrete grid; no common-cost interpolation', 'synthetic': cfg.backend.name == 'mock'})
    if plots:
        plot(out, curve_rows, cfg.backend.name == 'mock')
    return {'output': str(out), 'independent_estimates': len(raw), 'decisions': decisions}


def plot(out, rows, synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'iid': 'IID full rollout', 'uniform': 'Shared prefix + uniform',
              'entropy': 'Shared prefix + entropy', 'arithmetic': 'Arithmetic Sampling'}
    for metric, label in [('total_flops', 'Total operator FLOPs (including amortized setup)'),
                          ('gpu_seconds', 'GPU seconds (including amortized setup)')]:
        fig, ax = plt.subplots(figsize=(7.2, 4.7), layout='constrained')
        present = False
        for method in METHODS:
            points = [r for r in rows if r['method'] == method and r[metric] is not None]
            if not points:
                continue
            present = True
            points.sort(key=lambda r: r[metric])
            xs = np.asarray([r[metric] for r in points]); ys = np.asarray([r['js_bits'] for r in points])
            xerr = np.maximum(0, np.array([xs - [r[f'{metric}_ci_low'] for r in points], [r[f'{metric}_ci_high'] for r in points] - xs]))
            yerr = np.maximum(0, np.array([ys - [r['js_bits_ci_low'] for r in points], [r['js_bits_ci_high'] for r in points] - ys]))
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker='o', capsize=3, label=labels[method])
        ax.set(xlabel=label, ylabel='JS divergence to independent reference (bits)')
        ax.spines[['top', 'right']].set_visible(False)
        if present:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, 'GPU time has not been measured.\nCPU validation does not establish a speedup.',
                    ha='center', va='center', transform=ax.transAxes)
        if synthetic:
            ax.set_title('SYNTHETIC VALIDATION — no GSM8K conclusions', fontsize=10)
        fig.savefig(out / f'error_vs_{metric}.png', dpi=180)
        fig.savefig(out / f'error_vs_{metric}.pdf')
        plt.close(fig)
