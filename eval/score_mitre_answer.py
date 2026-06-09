#!/usr/bin/env python3
"""Score one MITRE answer with the deterministic coverage scorer."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_coverage_module() -> Any:
    path = ROOT / "eval" / "run_sft_coverage_suite.py"
    spec = importlib.util.spec_from_file_location("run_sft_coverage_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def read_text_arg(value: str, file_path: Path | None) -> str:
    if file_path is not None:
        return file_path.read_text()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-json", default="")
    parser.add_argument("--row-file", type=Path)
    parser.add_argument("--reply", default="")
    parser.add_argument("--reply-file", type=Path)
    parser.add_argument("--pass-threshold", type=float, default=0.67)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    row_source = read_text_arg(args.row_json, args.row_file)
    if not row_source.strip():
        raise ValueError("--row-json or --row-file is required")
    reply = read_text_arg(args.reply, args.reply_file)
    if not reply.strip():
        raise ValueError("--reply or --reply-file is required")

    row = json.loads(row_source)
    if not isinstance(row, dict):
        raise ValueError("row must be a JSON object")
    coverage = load_coverage_module()
    score = coverage.score_reply(row, reply, args.pass_threshold)
    payload = {
        "ok": True,
        "schema": "stoneytech.spark_mitre_answer_score.v1",
        "attack_id": str(row.get("attack_id") or ""),
        "kind": str(row.get("kind") or ""),
        "reply": reply,
        "score": score,
        "hard_gate_passed": bool(score.get("passed")),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
