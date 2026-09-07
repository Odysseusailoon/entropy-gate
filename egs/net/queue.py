"""Transactional shared task queue on local SQLite; no time-based sample cutoffs.

Each task is a matched block (question, budget, run) containing all four methods
in randomized order, so paired methods run on the SAME GPU. Five workers claim
blocks from one queue. Recovery is explicit and only reclaims a dead local PID.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import socket
import sqlite3
import time
from egs.io_utils import sha256_obj

class Queue:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'queue.sqlite', timeout=60, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA busy_timeout=60000')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, phase TEXT NOT NULL, ordinal INTEGER NOT NULL,
            payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
            worker TEXT, hostname TEXT, pid INTEGER, started REAL, finished REAL, error TEXT);
          CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
    def close(self):
        self.db.close()
    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
    def initialize(self, phase, jobs, fingerprint, seed):
        key = f'phase:{phase}'
        with self.transaction():
            old = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise RuntimeError('queue phase belongs to another frozen plan')
                return
            jobs = list(jobs)
            random.Random(seed).shuffle(jobs)
            for ordinal, job in enumerate(jobs):
                identity = sha256_obj([phase, job])
                self.db.execute('INSERT INTO tasks(id,phase,ordinal,payload) VALUES(?,?,?,?)',
                    (identity, phase, ordinal, json.dumps(job, sort_keys=True)))
            self.db.execute('INSERT INTO metadata VALUES(?,?)', (key, fingerprint))
    def claim(self, phase, worker):
        with self.transaction():
            failed = self.db.execute("SELECT id FROM tasks WHERE phase=? AND state='failed' LIMIT 1", (phase,)).fetchone()
            if failed:
                raise RuntimeError(f'queue halted by failed block {failed[0]}')
            row = self.db.execute("SELECT id,payload FROM tasks WHERE phase=? AND state='pending' AND (worker IS NULL OR worker=?) ORDER BY ordinal LIMIT 1", (phase, worker)).fetchone()
            if not row:
                return None
            self.db.execute("UPDATE tasks SET state='running',worker=?,hostname=?,pid=?,started=? WHERE id=?",
                (worker, socket.gethostname(), os.getpid(), time.time(), row[0]))
            return {'id': row[0], **json.loads(row[1])}
    def finish(self, task_id, worker, error=None):
        with self.transaction():
            cur = self.db.execute("UPDATE tasks SET state=?,finished=?,error=? WHERE id=? AND worker=? AND state='running'",
                ('failed' if error else 'done', time.time(), error, task_id, worker))
            if cur.rowcount != 1:
                raise RuntimeError('task ownership mismatch')
    def status(self, phase):
        return dict(self.db.execute('SELECT state,count(*) FROM tasks WHERE phase=? GROUP BY state', (phase,)))
    def recover_dead(self, phase):
        n = 0
        with self.transaction():
            for tid, host, pid in self.db.execute("SELECT id,hostname,pid FROM tasks WHERE phase=? AND state='running'", (phase,)).fetchall():
                if host != socket.gethostname():
                    continue
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    # Preserve worker/GPU affinity for an unfinished paired block.
                    self.db.execute("UPDATE tasks SET state='pending',hostname=NULL,pid=NULL,started=NULL WHERE id=?", (tid,))
                    n += 1
        return n
    def require_complete(self, phase):
        status = self.status(phase)
        if not status.get('done') or any(v for k, v in status.items() if k != 'done'):
            raise RuntimeError(f'{phase} queue incomplete: {status}')
