"""JSONL / atomic writes / run manifests. Nothing clever on purpose."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_str(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint(pkg_dir: str | Path) -> dict[str, str]:
    """Hash every .py in the package so a run can be tied to exact code.

    No git repo here, so this is the reproducibility anchor.
    """
    pkg_dir = Path(pkg_dir)
    out = {}
    for p in sorted(pkg_dir.rglob("*.py")):
        out[str(p.relative_to(pkg_dir.parent))] = sha256_file(p)
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=_default)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def build_manifest(cfg_dict: dict, extra: dict | None = None) -> dict:
    """Everything needed to say 'this number came from that code + that config'."""
    root = Path(__file__).resolve().parent
    man = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg_dict,
        "config_hash": sha256_obj(cfg_dict),
        "code_fingerprint": code_fingerprint(root),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "hostname": platform.node(),
    }
    try:
        man["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        man["nvidia_smi"] = None
    if extra:
        man.update(extra)
    return man
