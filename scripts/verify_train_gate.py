#!/usr/bin/env python3
"""Verify that a micro-smoke eval receipt is strong enough for long training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("summary JSON must be an object")
    return payload


def as_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nested = payload.get("summaries")
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict)]
    return [payload]


def validate_summary(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    total = as_int(payload, "total")
    passed = as_int(payload, "passed")
    errors = as_int(payload, "errors")
    full_budget_failures = as_int(payload, "full_budget_failures")

    if payload.get("ok") is not True:
        failures.append("ok_not_true")
    if total <= 0:
        failures.append("total_not_positive")
    if passed != total:
        failures.append("not_all_rows_passed")
    if errors != 0:
        failures.append("errors_nonzero")
    if full_budget_failures != 0:
        failures.append("full_budget_failures_nonzero")
    if payload.get("stopped_early"):
        failures.append("stopped_early")

    by_failure_type = payload.get("by_failure_type")
    if isinstance(by_failure_type, dict):
        non_passed = {
            str(key): value
            for key, value in by_failure_type.items()
            if str(key) != "passed" and as_int({"value": value}, "value") > 0
        }
        if non_passed:
            failures.append("non_passed_failure_types")
    return failures


def verify(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    receipts = summaries(payload)
    failures: list[dict[str, Any]] = []
    if not receipts:
        failures.append({"index": 0, "failures": ["missing_summaries"]})
    for index, receipt in enumerate(receipts):
        summary_failures = validate_summary(receipt)
        if summary_failures:
            failures.append(
                {
                    "index": index,
                    "summary": str(receipt.get("summary") or receipt.get("results") or path),
                    "failures": summary_failures,
                    "passed": as_int(receipt, "passed"),
                    "total": as_int(receipt, "total"),
                    "errors": as_int(receipt, "errors"),
                    "full_budget_failures": as_int(receipt, "full_budget_failures"),
                }
            )
    return {
        "ok": not failures,
        "summary": str(path),
        "checked_summaries": len(receipts),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    try:
        result = verify(args.summary)
    except Exception as exc:
        result = {"ok": False, "summary": str(args.summary), "failures": [{"failures": [str(exc)]}]}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
