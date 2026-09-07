"""Answer-quality checks and human review validation for frozen runs."""
import csv
from pathlib import Path
from .answer_parser import INVALID
from .io_utils import read_json, read_jsonl, sha256_file, write_json
from .metrics import TRUNCATED
from .protocol import protocol_id


def quality(answers, cfg):
    answers = list(answers)
    if not answers:
        raise ValueError("cannot validate an empty sample")
    invalid = answers.count(INVALID) / len(answers)
    truncated = answers.count(TRUNCATED) / len(answers)
    return {"n": len(answers), "invalid_rate": invalid, "truncation_rate": truncated,
            "passed": invalid < cfg.checks.invalid_threshold and truncated <= cfg.checks.truncation_threshold}


def audit(root, cfg):
    root = Path(root)
    records = {r["review_id"]: r for r in read_jsonl(root / "preflight_rollouts.jsonl")}
    if sha256_file(root / "preflight_rollouts.jsonl") != read_json(root / "preflight.json")["rollouts_sha256"]:
        raise RuntimeError("preflight rollouts were modified")
    with open(root / "manual_review.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for row in rows:
        key = row["review_id"]
        if key in seen or key not in records:
            raise ValueError("duplicate or unknown review id")
        seen.add(key)
        original = records[key]
        if row["text"] != original["text"] or row["parsed_answer"] != original["answer"] or row["question_id"] != original["question_id"]:
            raise ValueError("review must preserve original rollout and parser output")
        if row["approved"].strip().lower() != "yes" or row["human_answer"].strip() != original["answer"]:
            raise RuntimeError("every reviewed record needs approved=yes and a matching human_answer; parser disagreements must be resolved in a new run")
    if len(seen) < cfg.checks.manual_review_samples or seen != records.keys():
        raise RuntimeError("manual audit is incomplete")
    report = {"passed": True, "n_reviewed": len(seen), "protocol_id": protocol_id(root),
              "review_sha256": sha256_file(root / "manual_review.csv")}
    write_json(root / "manual_audit.json", report)
    return report
