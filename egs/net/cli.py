import argparse
import json
from .config import load_config
from .protocol import freeze, open_run
from .engine import Engine
from .pipeline import initialize, worker, calibrate, read_plan
from .reference import import_reference, initialize_extension, seal_reference
from .preflight import preflight
from .analysis import analyze
from .queue import Queue
from egs.checks import audit
from egs.protocol import require_ready


def main(argv=None):
    parser = argparse.ArgumentParser(description='End-to-end net benefit: IID / shared uniform / shared entropy / Arithmetic Sampling')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--config', action='append', required=True)
    p.add_argument('--offline', action='store_true', help='use cached artifacts at explicitly pinned SHA revisions')
    for name in ['preflight', 'audit', 'initialize', 'worker', 'calibrate', 'analyze', 'status', 'recover', 'import-reference', 'extend-reference', 'reference-check']:
        p = sub.add_parser(name); p.add_argument('--run', required=True)
        if name in ['initialize', 'worker', 'status', 'recover']:
            p.add_argument('--phase', required=True)
        if name == 'worker':
            p.add_argument('--worker', required=True)
        if name == 'import-reference':
            p.add_argument('--source', required=True)
        if name == 'extend-reference':
            p.add_argument('--samples', type=int, required=True)
        if name == 'analyze':
            p.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        print(freeze(load_config(*args.config), offline=args.offline)); return
    cfg, test, dev, fingerprint, manifest = open_run(args.run)
    if args.command == 'audit':
        result = audit(args.run, cfg)
    elif args.command == 'import-reference':
        result = import_reference(args.run, cfg, test, fingerprint, args.source)
    elif args.command == 'initialize':
        result = initialize(args.run, cfg, test, dev, args.phase)
    elif args.command == 'calibrate':
        require_ready(args.run, cfg)
        result = calibrate(args.run, cfg, test, dev)
    elif args.command == 'analyze':
        result = analyze(args.run, cfg, test, plots=not args.no_plots)
    elif args.command == 'reference-check':
        result = seal_reference(args.run, cfg, test)
    elif args.command == 'extend-reference':
        result = initialize_extension(args.run, cfg, test, args.samples)
    elif args.command in ['status', 'recover']:
        queue = Queue(args.run)
        result = queue.status(args.phase) if args.command == 'status' else {'recovered_dead_local_tasks': queue.recover_dead(args.phase)}
        queue.close()
    else:
        if args.command == 'worker':
            require_ready(args.run, cfg)
            if args.phase != 'calibration':
                read_plan(args.run)
            queue = Queue(args.run)
            if not queue.db.execute('SELECT 1 FROM metadata WHERE key=?', (f'phase:{args.phase}',)).fetchone():
                raise RuntimeError('initialize this queue phase before loading the model')
            queue.close()
        engine = Engine(cfg, fingerprint)
        result = preflight(args.run, cfg, dev, engine) if args.command == 'preflight' else worker(args.run, cfg, test, dev, engine, args.phase, args.worker)
    print(json.dumps(result, indent=2, ensure_ascii=False))
