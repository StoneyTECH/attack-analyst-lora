#!/usr/bin/env python3
"""Summarize MITRE eval failure types without rerunning inference."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ATTACK_ID_PATTERN = re.compile(r"\b(?:T\d{4}(?:\.\d{3})?|TA\d{4}|M\d{4})\b")
REPEATED_ATTACK_FRAGMENT_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3}){2,}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def attack_ids(text: str) -> list[str]:
    return sorted(set(ATTACK_ID_PATTERN.findall(text)))


def wrong_attack_ids(row: dict[str, Any]) -> list[str]:
    score = row.get("score") if isinstance(row.get("score"), dict) else {}
    diagnostics = score.get("diagnostics") if isinstance(score.get("diagnostics"), dict) else {}
    if diagnostics.get("wrong_attack_ids"):
        return list(diagnostics["wrong_attack_ids"])
    expected = str(row.get("attack_id") or "")
    return [item for item in attack_ids(str(row.get("reply") or "")) if item != expected]


def failure_type(row: dict[str, Any]) -> str:
    score = row.get("score") if isinstance(row.get("score"), dict) else {}
    diagnostics = score.get("diagnostics") if isinstance(score.get("diagnostics"), dict) else {}
    if diagnostics.get("failure_type"):
        return str(diagnostics["failure_type"])
    if score.get("passed"):
        return "passed"
    critical = set(score.get("critical_missing") or [])
    kind = str(row.get("kind") or "")
    reply = str(row.get("reply") or "")
    if row.get("error"):
        return "infrastructure_error"
    if REPEATED_ATTACK_FRAGMENT_PATTERN.search(reply):
        return "repetition_collapse"
    if kind == "fake_id_rejection":
        return "fake_id_acceptance"
    if kind == "procedure_to_technique" and {"attack_id", "technique_name"} & critical:
        return "procedure_disambiguation"
    if {"attack_id", "technique_name"} & critical:
        return "wrong_attack_mapping"
    if {"evidence_reasoning", "boundary_reasoning"} & critical:
        return "analyst_discipline"
    return "unknown_failure"


def summarize(rows: list[dict[str, Any]], limit_examples: int) -> dict[str, Any]:
    by_failure_type: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    for row in rows:
        ftype = failure_type(row)
        by_failure_type[ftype] = by_failure_type.get(ftype, 0) + 1
        kind = str(row.get("kind") or "unknown")
        bucket = by_kind.setdefault(kind, {"passed": 0, "failed": 0, "total": 0})
        bucket["total"] += 1
        if ftype == "passed":
            bucket["passed"] += 1
            continue
        bucket["failed"] += 1
        if len(examples) < limit_examples:
            score = row.get("score") if isinstance(row.get("score"), dict) else {}
            diagnostics = score.get("diagnostics") if isinstance(score.get("diagnostics"), dict) else {}
            examples.append(
                {
                    "attack_id": row.get("attack_id", ""),
                    "kind": kind,
                    "failure_type": ftype,
                    "critical_missing": score.get("critical_missing", []),
                    "wrong_attack_ids": wrong_attack_ids(row),
                    "max_new_tokens": row.get("max_new_tokens"),
                    "reply_excerpt": str(row.get("reply") or "")[:260],
                }
            )
    total = len(rows)
    failed = total - by_failure_type.get("passed", 0)
    return {
        "ok": True,
        "total": total,
        "passed": by_failure_type.get("passed", 0),
        "failed": failed,
        "failure_rate": round(failed / total, 4) if total else 0.0,
        "by_failure_type": by_failure_type,
        "by_kind": by_kind,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit-examples", type=int, default=20)
    args = parser.parse_args()

    payload = summarize(read_jsonl(args.results), args.limit_examples)
    payload["results"] = str(args.results)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        payload["summary"] = str(args.summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
