"""Run locks, provenance and readiness checks for the v3 protocol."""
from __future__ import annotations
from contextlib import contextmanager
from importlib.metadata import distributions, version
from pathlib import Path
import fcntl
from .io_utils import read_json, sha256_file


def runtime_versions():
    # A venv may inherit system packages. Resolve the winning distribution on
    # sys.path instead of letting a shadowed base package overwrite its version.
    names = {d.metadata["Name"].lower() for d in distributions()}
    return {name: version(name) for name in sorted(names)}


def protocol_sources():
    root = Path(__file__).resolve().parent.parent
    paths = [root / name for name in ("PREREGISTRATION.md", "pyproject.toml", "requirements.txt")]
    paths.extend(sorted((root / "scripts").glob("*.sh")))
    return {str(p.relative_to(root)): sha256_file(p) for p in paths if p.exists()}


@contextmanager
def role_lock(root, role):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f".{role}.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another {role} worker owns this run") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def protocol_id(root):
    return sha256_file(Path(root) / "manifest.json")


def require_ready(root, cfg):
    root = Path(root)
    path = root / "preflight.json"
    if not path.exists():
        raise RuntimeError("run preflight before any experiment worker")
    report = read_json(path)
    if not report["passed"] or report["protocol_id"] != protocol_id(root):
        raise RuntimeError("preflight failed or belongs to a different protocol")
    if cfg.checks.require_manual_review:
        audit_path = root / "manual_audit.json"
        if not audit_path.exists():
            raise RuntimeError("human review of at least 50 preflight rollouts is required; fill manual_review.csv then run audit")
        audit = read_json(audit_path)
        if not audit["passed"] or audit["protocol_id"] != protocol_id(root) or audit["review_sha256"] != sha256_file(root / "manual_review.csv"):
            raise RuntimeError("manual audit missing, stale or failed")
