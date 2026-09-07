from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import json
import pytest
from egs.io_utils import read_json, write_json, sha256_file
from egs.net.config import load_config
from egs.net.protocol import freeze, open_run
from egs.net.preflight import preflight
from egs.net.engine import Engine
from egs.net.pipeline import initialize, worker, calibrate, read_plan, result_path, load_result
from egs.net.reference import seal_reference
from egs.net.analysis import analyze
from egs.net.queue import Queue


def make_run(tmp_path):
    cfg = load_config('configs/net-demo.yaml', overrides={
        'run': {'out_dir': str(tmp_path)}, 'data': {'n_questions': 2},
        'study': {'replicates': 1, 'dev_questions': 1, 'bootstrap_samples': 50}})
    root = freeze(cfg)
    cfg, test, dev, fp, manifest = open_run(root)
    return root, cfg, test, dev, Engine(cfg, fp)


def test_shared_queue_claims_once_across_five_workers(tmp_path):
    q = Queue(tmp_path)
    jobs = [{'i': i} for i in range(100)]
    q.initialize('x', jobs, 'frozen', 99); q.close()
    def drain(worker_id):
        queue = Queue(tmp_path); got = []
        while (job := queue.claim('x', worker_id)) is not None:
            got.append(job['i']); queue.finish(job['id'], worker_id)
        queue.close(); return got
    with ThreadPoolExecutor(max_workers=5) as pool:
        batches = list(pool.map(drain, ['gpu0', 'gpu1', 'gpu2', 'gpu3', 'gpu4']))
    all_ids = [i for b in batches for i in b]
    assert sorted(all_ids) == list(range(100))
    q = Queue(tmp_path); q.require_complete('x'); q.close()


def test_queue_failed_block_halts_claims(tmp_path):
    q = Queue(tmp_path)
    q.initialize('x', [{'i': 1}, {'i': 2}], 'frozen', 1)
    job = q.claim('x', 'worker'); q.finish(job['id'], 'worker', 'FLOP coverage failed')
    with pytest.raises(RuntimeError, match='halted'):
        q.claim('x', 'other')
    q.close()


def test_net_pipeline_uses_frozen_dev_plan_and_actual_points(tmp_path):
    root, cfg, test, dev, engine = make_run(tmp_path)
    preflight(root, cfg, dev, engine)
    initialize(root, cfg, test, dev, 'calibration')
    worker(root, cfg, test, dev, engine, 'calibration', 'cpu0')
    plan = calibrate(root, cfg, test, dev)
    assert set(plan['development_ids']).isdisjoint(plan['held_out_ids'])
    assert plan['sample_counts']['iid'] == {'16': 16, '32': 32, '64': 64}
    original = sha256_file(root / 'calibration_plan.json')
    initialize(root, cfg, test, dev, 'test')
    worker(root, cfg, test, dev, engine, 'test', 'cpu0')
    assert sha256_file(root / 'calibration_plan.json') == original
    result = analyze(root, cfg, test, plots=False)
    assert result['independent_estimates'] == 2 * 3 * 1 * 4
    decision = read_json(root / 'net_analysis/decision.json')
    assert 'no common-cost interpolation' in decision['threshold_rule']
    assert decision['synthetic']
    assert (root / 'net_analysis/speedups.csv').exists()
    for q in test:
        for b in cfg.study.iid_budgets:
            for method in ['iid', 'uniform', 'entropy', 'arithmetic']:
                row = load_result(result_path(root, 'test', q.question_id, b, 0, method))
                assert row['samples_requested'] == plan['sample_counts'][method][str(b)]
    # A mutated plan cannot be silently applied to testing.
    path = root / 'calibration_plan.json'; data = read_json(path); data['beta'] = 9
    write_json(path, data)
    with pytest.raises(RuntimeError, match='changed'):
        read_plan(root)


def test_unseen_set_and_promotion_gate(tmp_path):
    root, cfg, test, dev, engine = make_run(tmp_path)
    with pytest.raises(RuntimeError, match='preflight'):
        initialize(root, cfg, test, dev, 'calibration')
    with pytest.raises(RuntimeError, match='overlaps'):
        freeze(load_config('configs/net-demo.yaml', overrides={'run': {'name': 'overlap', 'out_dir': str(tmp_path)},
            'study': {'tuning_question_ids': ['toy-1000']}}))


@pytest.mark.parametrize('failure', ['invalid', 'truncated'])
def test_calibration_rejects_failed_development_quality_before_sealing(tmp_path, failure):
    from egs.metrics import INVALID, TRUNCATED
    from egs.net.pipeline import save_result
    root, cfg, test, dev, engine = make_run(tmp_path)
    preflight(root, cfg, dev, engine)
    initialize(root, cfg, test, dev, 'calibration')
    worker(root, cfg, test, dev, engine, 'calibration', 'cpu0')
    paths = [result_path(root, 'calibration', q.question_id, n, rep, 'entropy')
             for q in dev for n in cfg.study.candidates for rep in range(cfg.study.dev_replicates)]
    originals = {path: load_result(path) for path in paths}
    # One rare censored run must fail the zero-truncation rule. Invalid answers
    # use the preregistered aggregate threshold, rather than a per-leaf test.
    changed = paths[:1] if failure == 'truncated' else paths
    for path in changed:
        row = dict(originals[path])
        row['distribution'] = {TRUNCATED if failure == 'truncated' else INVALID: 1.}
        save_result(path, row)
    preserved = {path: sha256_file(path) for path in paths}
    with pytest.raises(RuntimeError, match='entropy calibration invalid/truncation'):
        calibrate(root, cfg, test, dev)
    assert not (root / 'calibration_plan.json').exists()
    assert not (root / 'calibration_seal.json').exists()
    assert all(sha256_file(path) == digest for path, digest in preserved.items())
    # Restore this artificial test fixture; real failed samples are never
    # removed or rerolled to make a frozen experiment pass.
    for path, row in originals.items():
        save_result(path, row)
    plan = calibrate(root, cfg, test, dev)
    assert plan['development_quality']['entropy']['truncation_rate'] == 0


def test_reference_import_validates_sealed_raw_samples(tmp_path):
    from egs.net.reference import import_reference
    from egs.net.pipeline import save_result
    source, cfg, qs, dev, engine = make_run(tmp_path / 'source')
    for q in qs:
        save_result(source / 'reference_chunks' / q.question_id / '000000.json',
            {'start': 0, 'samples': [{'answer': str(i % 2)} for i in range(128)]})
    seal_reference(source, cfg, qs)
    target_cfg = load_config('configs/net-demo.yaml', overrides={
        'run': {'out_dir': str(tmp_path / 'target')}, 'data': {'n_questions': 2},
        'sampling': {'seed_base': cfg.sampling.seed_base + 1}})
    target = freeze(target_cfg)
    tc, tq, _, fp, _ = open_run(target)
    accepted = import_reference(target, tc, tq, fp, source)
    assert accepted['passed'] and all(n == 128 for n in accepted['n_by_question'].values())
    # Even a correctly re-checksummed raw checkpoint must match its ORIGINAL seal.
    changed = source / 'reference_chunks' / qs[0].question_id / '000000.json'
    row = load_result(changed); row['samples'][0]['answer'] = 'edited'
    save_result(changed, row)
    with pytest.raises(RuntimeError, match='modified after sealing'):
        import_reference(target, tc, tq, fp, source)


def test_reference_extension_cannot_overlap_initial_or_extension_jobs(tmp_path, monkeypatch):
    from egs.net.reference import initialize_extension
    from egs.net.pipeline import save_result
    root, cfg, qs, _, _ = make_run(tmp_path)
    monkeypatch.setattr('egs.net.pipeline.read_plan', lambda root: {})
    queue = Queue(root); queue.initialize('test', [{'kind': 'reference'}], 'test', 1)
    with pytest.raises(RuntimeError, match='incomplete'):
        initialize_extension(root, cfg, qs, 256)
    task = queue.claim('test', 'cpu0'); queue.finish(task['id'], 'cpu0')
    for q in qs:
        save_result(root / 'reference_chunks' / q.question_id / '000000.json',
            {'start': 0, 'samples': [{'answer': '1'}] * 128})
    phase = initialize_extension(root, cfg, qs, 256)
    task = queue.claim(phase, 'cpu0')
    assert task['start'] == 128 and task['samples'] == 128
    with pytest.raises(RuntimeError, match='existing reference extension'):
        initialize_extension(root, cfg, qs, 256)
    queue.close()


def test_dead_worker_recovery_keeps_gpu_affinity(tmp_path, monkeypatch):
    queue = Queue(tmp_path); queue.initialize('test', [{'block': 1}], 'frozen', 9)
    task = queue.claim('test', 'gpu2')
    def dead(*args):
        raise ProcessLookupError()
    monkeypatch.setattr('egs.net.queue.os.kill', dead)
    assert queue.recover_dead('test') == 1
    assert queue.claim('test', 'gpu0') is None
    assert queue.claim('test', 'gpu2')['id'] == task['id']
    queue.close()


def test_shared_gpu_admission_rejects_occupied_cards():
    from egs.net.devices import validate_devices
    cards = [['0', 'gpu-a', 'NVIDIA H200', '0', '0'], ['1', 'gpu-b', 'NVIDIA H200', '0', '0'],
             ['2', 'gpu-c', 'NVIDIA A100', '0', '0']]
    assert validate_devices(cards, [['gpu-b', '123']], ['0'])[0]['uuid'] == 'gpu-a'
    with pytest.raises(RuntimeError, match='occupied'):
        validate_devices(cards, [['gpu-b', '123']], ['1'])
    with pytest.raises(RuntimeError, match='requires H200'):
        validate_devices(cards, [], ['2'])


def test_audit_requires_actual_review_fields(tmp_path):
    from egs.checks import audit
    from egs.protocol import require_ready
    root, cfg, _, dev, engine = make_run(tmp_path)
    preflight(root, cfg, dev, engine)
    cfg.checks.require_manual_review = True
    with pytest.raises(RuntimeError, match='human review'):
        require_ready(root, cfg)
    with pytest.raises(RuntimeError, match='approved=yes'):
        audit(root, cfg)
    # Synthetic test fixture only: exercise the human-review validation path.
    path = root / 'manual_review.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['approved'] = 'yes'
        row['human_answer'] = row['parsed_answer']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert audit(root, cfg)['n_reviewed'] == 50
    require_ready(root, cfg)


def test_frozen_files_and_unknown_keys(tmp_path):
    root, cfg, _, _, _ = make_run(tmp_path)
    with pytest.raises(RuntimeError, match='already frozen'):
        freeze(cfg)
    with (root / 'questions.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(RuntimeError, match='frozen file changed'):
        open_run(root)
    with pytest.raises(KeyError):
        load_config(overrides={'study': {'flopps': 100}})


def test_cli_checks_gates_before_loading_model(tmp_path, monkeypatch):
    from egs.net import cli
    root, _, _, _, _ = make_run(tmp_path)
    def forbidden(*args):
        raise AssertionError('model must not load before the promotion gates pass')
    monkeypatch.setattr(cli, 'Engine', forbidden)
    with pytest.raises(RuntimeError, match='preflight'):
        cli.main(['worker', '--run', str(root), '--phase', 'test', '--worker', 'cpu0'])


def test_reference_import_rejects_unsupported_protocol(tmp_path):
    from egs.net.reference import import_reference
    root, cfg, qs, _, _ = make_run(tmp_path)
    source = tmp_path / 'unsupported-source'
    write_json(source / 'manifest.json', {'protocol': 'v2'})
    with pytest.raises(RuntimeError, match='sealed v3 source'):
        import_reference(root, cfg, qs, {}, source)


def test_runtime_fingerprint_uses_active_venv_package(monkeypatch):
    from types import SimpleNamespace
    from egs.protocol import runtime_versions
    monkeypatch.setattr('egs.protocol.distributions', lambda: [
        SimpleNamespace(metadata={'Name': 'Transformers'}, version='4.57.6'),
        SimpleNamespace(metadata={'Name': 'Transformers'}, version='5.6.2')])
    monkeypatch.setattr('egs.protocol.version', lambda name: '4.57.6')
    assert runtime_versions() == {'transformers': '4.57.6'}


def test_amended_profile_keeps_experiment_rules():
    c = load_config('configs/net-benefit.yaml', 'configs/net-qwen17b-a100.yaml')
    assert c.model.name == 'Qwen/Qwen3-1.7B' and c.study.required_gpu_name == 'A100'
    assert c.data.n_questions == 50 and c.study.replicates == 10
    assert c.study.iid_budgets == [16, 32, 64] and c.reference.n_ref == 1024
    assert c.checks.require_manual_review and c.checks.truncation_threshold == 0
