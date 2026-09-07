"""Isolated CUDA compatibility probe; never accepted as a preregistered study.

Uses development prompts only, completes predetermined paths, records actual
FLOPs and occupied GPU seconds, and never terminates an existing GPU process.
"""
import argparse
import os
from pathlib import Path
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--questions', required=True)
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', default='0', help='physical GPU index; must be idle')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--methods', nargs='+', default=['iid', 'uniform', 'entropy', 'arithmetic'],
                        choices=['iid', 'uniform', 'entropy', 'arithmetic'])
    args = parser.parse_args()
    from .devices import inventory, validate_devices
    from egs.protocol import role_lock
    card = validate_devices(*inventory(), [args.gpu], required_name='')[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = card['uuid']
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    with role_lock('/tmp', f"entropy-gate-{card['uuid']}"):
        from .config import load_config
        from .engine import Engine
        from .experiment import measure
        from .arithmetic import Sampler
        from .pipeline import save_result
        from egs.io_utils import read_json, read_jsonl, write_json, code_fingerprint
        from egs.data import Question
        from egs.backend import seed_for
        from egs.metrics import INVALID, TRUNCATED
        import torch
        from transformers import AutoTokenizer
        from importlib.metadata import version
        torch.set_num_threads(1)
        cfg = load_config('configs/net-benefit.yaml', overrides={
            'model': {'name': args.model, 'revision': args.revision,
                      'tokenizer_name': args.model, 'tokenizer_revision': args.revision},
            'backend': {'device': 'cuda:0', 'batch_size': args.batch_size},
            'sampling': {'max_new_tokens': args.max_tokens}})
        if args.samples < cfg.study.roots * cfg.study.n_min:
            raise ValueError('samples must cover every fixed root')
        questions = [Question(**r) for r in read_jsonl(args.questions)]
        if any(not q.question_id.startswith('train-') or q.index < 8 for q in questions):
            raise ValueError('debug must use held-out development train questions, excluding demonstrations')
        q = questions[0]
        fp = read_json(args.fingerprint)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        if tokenizer(q.prompt_text, add_special_tokens=False)['input_ids'] != q.input_ids:
            raise RuntimeError('model tokenizer does not match the frozen development prompt')
        out = Path(args.out)
        if (out / 'debug_session.json').exists():
            raise RuntimeError('use a fresh debug output directory')
        metadata = {'debug_only': True, 'counts_as_formal_experiment': False, 'pid': os.getpid(),
            'hostname': socket.gethostname(), 'gpu': card, 'started': time.time(), 'config': cfg.to_dict(),
            'question_id': q.question_id, 'code_fingerprint': code_fingerprint(Path(__file__).resolve().parents[1]),
            'runtime': {p: version(p) for p in ['torch', 'transformers', 'tokenizers', 'numpy']}}
        write_json(out / 'debug_session.json', metadata)
        print(f"Loading {args.model} on idle GPU {args.gpu} ({card['name']}); PID {os.getpid()}", flush=True)
        engine = Engine(cfg, fp, tokenizer=tokenizer)
        # Warm kernels without choosing test outcomes or any gate parameter.
        with torch.inference_mode():
            state = engine.prefill(q)
            engine.advance([state], Sampler('iid', 1, 49281, engine.device), stop_at=4)
        engine.synchronize()
        summary = []
        for method in args.methods:
            family = 'fork' if method in {'uniform', 'entropy'} else method
            print(f'Starting {method}: {args.samples} predetermined paths', flush=True)
            result = measure(engine, q, method, args.samples, seed_for(710283, family))
            for row in result['samples']:
                row['text'] = engine.decode(row['token_ids'])
            result.update({'debug_only': True, 'question_id': q.question_id, 'hardware': card})
            save_result(out / f'{method}.json', result)
            entry = {'method': method, 'gpu_seconds': result['cost']['gpu_seconds'],
                'total_flops': result['cost']['total_flops'], 'distribution': result['distribution'],
                'prompt_prefills': result['ledger']['prompt_prefills'],
                'invalid_mass': result['distribution'].get(INVALID, 0),
                'truncated_mass': result['distribution'].get(TRUNCATED, 0),
                'replay_verified': result['cost']['instrumentation_replay_verified']}
            summary.append(entry)
            write_json(out / 'summary.json', {'debug_only': True, 'gpu': card, 'methods': summary})
            print(entry, flush=True)
        write_json(out / 'debug_complete.json', {'debug_only': True, 'completed': time.time(), 'methods': args.methods})


if __name__ == '__main__':
    main()
