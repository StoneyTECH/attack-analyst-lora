#!/usr/bin/env python3
"""Run broad MITRE SFT coverage evals against a local chat endpoint."""
from __future__ import annotations

import argparse
import json
import re
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are a defensive cyber analyst. Use MITRE ATT&CK facts only when the "
    "evidence supports them. Prefer technique IDs, name missing evidence, and "
    "do not invent ATT&CK objects."
)

EVIDENCE_TERMS = [
    "evidence",
    "telemetry",
    "log",
    "logs",
    "event",
    "signal",
    "observed",
    "supports",
]

BOUNDARY_TERMS = [
    "do not claim",
    "not claim",
    "should not infer",
    "not infer",
    "cannot infer",
    "attribution",
    "uncertain",
    "missing evidence",
    "from this alone",
]

REJECTION_TERMS = [
    "cannot verify",
    "not valid",
    "not a valid",
    "invalid",
    "no official",
    "not official",
    "unknown",
    "not found",
    "not in the provided corpus",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

ATTACK_ID_PATTERN = re.compile(r"(?<![A-Z0-9])(?:T\d{4}(?:\.\d{3})?|TA\d{4}|M\d{4})(?![A-Z0-9.])", re.IGNORECASE)
REPEATED_ATTACK_FRAGMENT_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3}){2,}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def request_json(url: str, payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def contains_attack_id(text: str, attack_id: str) -> bool:
    return attack_id in attack_ids(text)


def allowed_wrong_id_contrast(reply: str, wrong_id: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", reply):
        if wrong_id not in attack_ids(sentence):
            continue
        normalized = normalize(sentence)
        if any(
            phrase in normalized
            for phrase in [
                "do not use",
                "do not map",
                "not supported",
                "is not supported",
                "wrong id",
                "wrong mapping",
                "unless the evidence",
                "do not substitute",
            ]
        ):
            return True
    return False


def unadjudicated_wrong_attack_ids(reply: str, expected_attack_id: str) -> list[str]:
    wrong_ids = [item for item in attack_ids(reply) if item != expected_attack_id]
    return [item for item in wrong_ids if not allowed_wrong_id_contrast(reply, item)]


def significant_words(value: str) -> list[str]:
    return [word for word in normalize(value).split() if len(word) >= 3 and word not in STOPWORDS]


def word_coverage(text: str, expected: str) -> float:
    words = significant_words(expected)
    if not words:
        return 1.0
    normalized = set(normalize(text).split())
    hits = sum(1 for word in words if word in normalized)
    return hits / len(words)


def attack_ids(text: str) -> list[str]:
    return sorted({item.upper() for item in ATTACK_ID_PATTERN.findall(text)})


def repeated_attack_fragments(text: str) -> list[str]:
    return sorted(set(REPEATED_ATTACK_FRAGMENT_PATTERN.findall(text)))


def assistant_answer(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if role in {"system", "user"} and content:
            messages.append({"role": role, "content": content})
    if not any(message["role"] == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError(f"row {row.get('attack_id') or row.get('id') or '<unknown>'} has no user prompt")
    return messages


def expected_name(row: dict[str, Any], expected: str) -> str:
    attack_id = str(row.get("attack_id") or "")
    if not attack_id:
        return ""
    patterns = [
        rf"{re.escape(attack_id)}\s+is\s+(.+?)\.",
        rf"Technique:\s*{re.escape(attack_id)}\s+(.+?)\.",
        rf"ATT&CK mapping:\s*{re.escape(attack_id)}\s+(.+?)\.",
        rf"Allowed mapping:\s*{re.escape(attack_id)}\s+(.+?)(?:,|\.|\n)",
        rf"Best supported ATT&CK mapping from the provided corpus:\s*{re.escape(attack_id)}\s+(.+?)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, expected, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
    return ""


def expected_line_values(expected: str, label: str) -> list[str]:
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)\.", expected, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return []
    value = match.group(1).strip()
    if not value or value.lower().startswith("not listed"):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def component(name: str, passed: bool, weight: float = 1.0, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": passed, "weight": weight, "detail": detail}


def score_fake_id(row: dict[str, Any], reply: str, threshold: float) -> dict[str, Any]:
    attack_id = str(row.get("attack_id") or "")
    normalized = normalize(reply)
    comps = [
        component("fake_id_mentioned", contains_attack_id(reply, attack_id), 1.0, attack_id),
        component("rejection_language", any(normalize(term) in normalized for term in REJECTION_TERMS), 2.0),
        component("no_validity_claim", "valid mitre attack" not in normalized and "official technique" not in normalized, 1.0),
    ]
    return summarize_score(comps, ["rejection_language"], threshold)


def score_technique(row: dict[str, Any], reply: str, threshold: float) -> dict[str, Any]:
    expected = assistant_answer(row)
    attack_id = str(row.get("attack_id") or "")
    name = expected_name(row, expected)
    tactics = expected_line_values(expected, "Tactics")
    platforms = expected_line_values(expected, "Platforms")
    normalized = normalize(reply)
    wrong_ids = unadjudicated_wrong_attack_ids(reply, attack_id)

    comps = [
        component("attack_id", contains_attack_id(reply, attack_id), 2.0, attack_id),
        component("technique_name", bool(name) and word_coverage(reply, name) >= 0.67, 2.0, name),
        component("evidence_reasoning", any(normalize(term) in normalized for term in EVIDENCE_TERMS), 1.0),
        component("boundary_reasoning", any(normalize(term) in normalized for term in BOUNDARY_TERMS), 1.0),
        component("no_unadjudicated_wrong_attack_ids", not wrong_ids, 2.0, wrong_ids),
    ]
    if tactics:
        comps.append(component("tactic_coverage", any(word_coverage(reply, tactic) >= 0.67 for tactic in tactics), 1.0, tactics))
    if platforms:
        comps.append(component("platform_coverage", any(word_coverage(reply, platform) >= 0.67 for platform in platforms), 1.0, platforms))
    critical = ["attack_id", "technique_name", "no_unadjudicated_wrong_attack_ids"]
    if row.get("require_discipline") is not False:
        critical.extend(["evidence_reasoning", "boundary_reasoning"])
    return summarize_score(comps, critical, threshold)


def summarize_score(components: list[dict[str, Any]], critical_names: list[str], threshold: float) -> dict[str, Any]:
    total_weight = sum(float(item["weight"]) for item in components)
    passed_weight = sum(float(item["weight"]) for item in components if item["passed"])
    component_score = passed_weight / total_weight if total_weight else 0.0
    missing = [item["name"] for item in components if not item["passed"]]
    critical_missing = [name for name in missing if name in critical_names]
    return {
        "passed": not critical_missing and component_score >= threshold,
        "component_score": round(component_score, 4),
        "missing": missing,
        "critical_missing": critical_missing,
        "components": components,
        "threshold": threshold,
    }


def classify_failure(row: dict[str, Any], reply: str, score: dict[str, Any]) -> dict[str, Any]:
    expected_attack_id = str(row.get("attack_id") or "")
    detected_ids = attack_ids(reply)
    wrong_ids = [item for item in detected_ids if item != expected_attack_id]
    unadjudicated_wrong_ids = unadjudicated_wrong_attack_ids(reply, expected_attack_id)
    repeated_fragments = repeated_attack_fragments(reply)
    critical_missing = set(score.get("critical_missing") or [])
    missing = set(score.get("missing") or [])
    kind = str(row.get("kind") or "unknown")

    if repeated_fragments:
        failure_type = "repetition_collapse"
    elif kind == "procedure_to_technique" and unadjudicated_wrong_ids:
        failure_type = "procedure_disambiguation"
    elif unadjudicated_wrong_ids and "no_unadjudicated_wrong_attack_ids" in critical_missing:
        failure_type = "wrong_attack_mapping"
    elif score.get("passed"):
        failure_type = "passed"
    elif "request_error" in critical_missing:
        failure_type = "infrastructure_error"
    elif kind == "fake_id_rejection":
        failure_type = "fake_id_acceptance"
    elif kind == "procedure_to_technique" and {"attack_id", "technique_name"} & critical_missing:
        failure_type = "procedure_disambiguation"
    elif {"attack_id", "technique_name"} & critical_missing:
        failure_type = "wrong_attack_mapping" if wrong_ids else "missing_attack_mapping"
    elif {"evidence_reasoning", "boundary_reasoning"} & critical_missing:
        failure_type = "analyst_discipline"
    elif missing:
        failure_type = "coverage_gap"
    else:
        failure_type = "unknown_failure"

    return {
        "failure_type": failure_type,
        "detected_attack_ids": detected_ids,
        "wrong_attack_ids": wrong_ids,
        "unadjudicated_wrong_attack_ids": unadjudicated_wrong_ids,
        "repeated_attack_fragments": repeated_fragments,
    }


def score_reply(row: dict[str, Any], reply: str, threshold: float = 0.67) -> dict[str, Any]:
    if row.get("kind") == "fake_id_rejection":
        score = score_fake_id(row, reply, threshold)
    else:
        score = score_technique(row, reply, threshold)
    score["diagnostics"] = classify_failure(row, reply, score)
    return score


def simulate_answer(row: dict[str, Any]) -> str:
    expected = assistant_answer(row)
    if expected:
        return expected
    attack_id = str(row.get("attack_id") or "unknown")
    return f"{attack_id} simulated answer. Evidence and boundaries should be named."


def error_result(row: dict[str, Any], budget: int, started: float, exc: Exception, threshold: float) -> dict[str, Any]:
    return {
        "attack_id": row.get("attack_id", ""),
        "kind": row.get("kind", "unknown"),
        "prompt": "",
        "reply": "",
        "score": {
            "passed": False,
            "component_score": 0.0,
            "missing": ["request_error"],
            "critical_missing": ["request_error"],
            "components": [],
            "threshold": threshold,
            "diagnostics": {
                "failure_type": "infrastructure_error",
                "detected_attack_ids": [],
                "wrong_attack_ids": [],
                "repeated_attack_fragments": [],
            },
        },
        "model": {},
        "seconds": round(time.time() - started, 3),
        "max_new_tokens": budget,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_row(args: argparse.Namespace, row: dict[str, Any], budget: int) -> dict[str, Any]:
    started = time.time()
    messages = prompt_messages(row)
    if args.simulate:
        reply = simulate_answer(row)
        model = {"simulated": True}
    else:
        response = request_json(
            args.chat_url,
            {
                "messages": messages,
                "max_new_tokens": budget,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
            },
            timeout=args.request_timeout_seconds,
        )
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "chat endpoint failed")
        reply = response.get("reply") or ""
        model = response.get("model") or {}
    return {
        "attack_id": row.get("attack_id", ""),
        "kind": row.get("kind", "unknown"),
        "prompt": next(message["content"] for message in messages if message["role"] == "user"),
        "reply": reply,
        "score": score_reply(row, reply, args.pass_threshold),
        "model": model,
        "seconds": round(time.time() - started, 3),
        "max_new_tokens": budget,
    }


def output_for_budget(path: Path, budget: int, multi_budget: bool) -> Path:
    if not multi_budget:
        return path
    return path.with_name(f"{path.stem}-tok{budget}{path.suffix}")


def estimated_budget(row: dict[str, Any], budgets: list[int]) -> int:
    expected_words = len(assistant_answer(row).split())
    kind = str(row.get("kind") or "")
    if kind == "fake_id_rejection":
        target = 128
    elif kind == "technique_concise_card":
        target = 128 if expected_words <= 90 else 256
    elif expected_words <= 90:
        target = 128
    elif expected_words <= 180:
        target = 256
    elif expected_words <= 380:
        target = 512
    elif expected_words <= 760:
        target = 1024
    else:
        target = 2048
    for budget in sorted(budgets):
        if budget >= target:
            return budget
    return max(budgets)


def adaptive_budgets_for_row(row: dict[str, Any], budgets: list[int], start_policy: str) -> list[int]:
    ordered = sorted(dict.fromkeys(budgets))
    if start_policy == "minimum":
        return ordered
    start = estimated_budget(row, ordered)
    start_index = ordered.index(start)
    return ordered[start_index:]


def compact_attempt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": result["max_new_tokens"],
        "seconds": result["seconds"],
        "passed": result["score"]["passed"],
        "component_score": result["score"]["component_score"],
        "missing": result["score"].get("missing", []),
        "critical_missing": result["score"].get("critical_missing", []),
        "diagnostics": result["score"].get("diagnostics", {}),
        "error": result.get("error", ""),
        "reply": result.get("reply", ""),
    }


def append_attempt(args: argparse.Namespace, row: dict[str, Any], attempt: dict[str, Any], attempt_index: int) -> None:
    if not args.attempts_out:
        return
    args.attempts_out.parent.mkdir(parents=True, exist_ok=True)
    trace = {
        "attack_id": row.get("attack_id", ""),
        "kind": row.get("kind", "unknown"),
        "attempt_index": attempt_index,
        **attempt,
    }
    with args.attempts_out.open("a") as handle:
        handle.write(json.dumps(trace, sort_keys=True) + "\n")


def run_adaptive_row(args: argparse.Namespace, row: dict[str, Any], budgets: list[int]) -> dict[str, Any]:
    attempts = []
    last_result: dict[str, Any] | None = None
    for budget in adaptive_budgets_for_row(row, budgets, args.adaptive_start_policy):
        started = time.time()
        try:
            result = run_row(args, row, budget)
        except Exception as exc:
            result = error_result(row, budget, started, exc, args.pass_threshold)
        attempt = compact_attempt(result)
        attempts.append(attempt)
        append_attempt(args, row, attempt, len(attempts))
        last_result = result
        if result["score"]["passed"]:
            break
    if last_result is None:
        raise RuntimeError("adaptive budget list is empty")
    last_result["adaptive"] = True
    last_result["attempts"] = attempts
    last_result["attempt_count"] = len(attempts)
    last_result["budget_path"] = [attempt["max_new_tokens"] for attempt in attempts]
    return last_result


def summarize_results(args: argparse.Namespace, suite: Path, results: list[dict[str, Any]], budget: int, out: Path) -> dict[str, Any]:
    passed = sum(1 for result in results if result["score"]["passed"])
    errors = sum(1 for result in results if result.get("error"))
    full_budget_failures = sum(1 for result in results if result.get("full_budget_failure"))
    by_kind: dict[str, dict[str, int]] = {}
    by_failure_type: dict[str, int] = {}
    for result in results:
        bucket = by_kind.setdefault(result["kind"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        if result["score"]["passed"]:
            bucket["passed"] += 1
        failure_type = str((result["score"].get("diagnostics") or {}).get("failure_type") or "unknown")
        by_failure_type[failure_type] = by_failure_type.get(failure_type, 0) + 1
    component_scores = [float(result["score"]["component_score"]) for result in results]
    budget_counts: dict[str, int] = {}
    attempt_counts = 0
    for result in results:
        budget_key = str(result.get("max_new_tokens", budget))
        budget_counts[budget_key] = budget_counts.get(budget_key, 0) + 1
        attempt_counts += int(result.get("attempt_count", 1))
    return {
        "ok": True,
        "suite": str(suite),
        "results": str(out),
        "total": len(results),
        "passed": passed,
        "errors": errors,
        "full_budget_failures": full_budget_failures,
        "accuracy": round(passed / len(results), 4) if results else 0.0,
        "avg_component_score": round(statistics.mean(component_scores), 4) if component_scores else 0.0,
        "min_component_score": round(min(component_scores), 4) if component_scores else 0.0,
        "max_new_tokens": budget,
        "budget_counts": budget_counts,
        "attempt_count": attempt_counts,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "pass_threshold": args.pass_threshold,
        "by_kind": by_kind,
        "by_failure_type": by_failure_type,
        "simulated": args.simulate,
        "adaptive": bool(getattr(args, "adaptive_token_budgets", "")),
        "adaptive_start_policy": getattr(args, "adaptive_start_policy", ""),
        "kinds": getattr(args, "kinds", ""),
        "max_rows_per_kind": int(getattr(args, "max_rows_per_kind", 0)),
        "stopped_early": bool(getattr(args, "stopped_early", False)),
        "stop_reason": getattr(args, "stop_reason", ""),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_budget(args: argparse.Namespace, rows: list[dict[str, Any]], budget: int, out: Path, summary_path: Path) -> dict[str, Any]:
    results = []
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in rows:
            started = time.time()
            try:
                result = run_row(args, row, budget)
            except Exception as exc:
                result = error_result(row, budget, started, exc, args.pass_threshold)
            results.append(result)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
    summary = summarize_results(args, args.suite, results, budget, out)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_adaptive(args: argparse.Namespace, rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    results = []
    full_budget_failures = 0
    max_budget = max(budgets)
    args.stopped_early = False
    args.stop_reason = ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if args.attempts_out is None:
        args.attempts_out = args.out.with_name(f"{args.out.stem}-attempts{args.out.suffix}")
    if args.attempts_out.exists():
        args.attempts_out.unlink()
    with args.out.open("w") as handle:
        for row in rows:
            result = run_adaptive_row(args, row, budgets)
            result["full_budget_failure"] = (
                not result["score"]["passed"]
                and int(result.get("max_new_tokens", 0)) == max_budget
                and not result.get("error")
            )
            if result["full_budget_failure"]:
                full_budget_failures += 1
            results.append(result)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if args.stop_after_full_budget_failures and full_budget_failures >= args.stop_after_full_budget_failures:
                args.stopped_early = True
                args.stop_reason = (
                    f"stopped after {full_budget_failures} adaptive rows failed at the "
                    f"{max_budget}-token ceiling"
                )
                break
    summary = summarize_results(args, args.suite, results, max_budget, args.out)
    summary["adaptive_token_budgets"] = sorted(dict.fromkeys(budgets))
    summary["attempts_out"] = str(args.attempts_out)
    summary["requested_rows"] = len(rows)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_budgets(value: str, fallback: int) -> list[int]:
    if not value.strip():
        return [fallback]
    budgets = []
    for item in value.split(","):
        item = item.strip()
        if item:
            budgets.append(int(item))
    if not budgets:
        raise ValueError("token budget list is empty")
    return budgets


def sample_rows(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if not sample_size or sample_size >= len(rows):
        return rows
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind") or "unknown"), []).append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bucket in by_kind.values():
        selected.append(rng.choice(bucket))
    remaining = [row for row in rows if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return selected[:sample_size]


def parse_kinds(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def filter_rows(rows: list[dict[str, Any]], kinds: set[str], max_rows_per_kind: int) -> list[dict[str, Any]]:
    if kinds:
        rows = [row for row in rows if str(row.get("kind") or "unknown") in kinds]
    if max_rows_per_kind:
        counts: dict[str, int] = {}
        capped = []
        for row in rows:
            kind = str(row.get("kind") or "unknown")
            count = counts.get(kind, 0)
            if count >= max_rows_per_kind:
                continue
            counts[kind] = count + 1
            capped.append(row)
        rows = capped
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=Path("artifacts/data/mitre_eval.jsonl"), type=Path)
    parser.add_argument("--chat-url", default="http://127.0.0.1:18080/chat")
    parser.add_argument("--out", default=Path("artifacts/evals/mitre_sft_coverage_results.jsonl"), type=Path)
    parser.add_argument("--summary", default=Path("artifacts/evals/mitre_sft_coverage_summary.json"), type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--token-budgets", default="", help="Comma-separated max_new_tokens ladder, for example 2048,1024,520.")
    parser.add_argument("--adaptive-token-budgets", default="", help="Comma-separated retry ladder per row, for example 128,256,512,1024,2048.")
    parser.add_argument("--adaptive-start-policy", choices=["minimum", "estimated"], default="minimum")
    parser.add_argument("--stop-after-full-budget-failures", type=int, default=0, help="In adaptive mode, stop after this many rows fail even at the largest token budget.")
    parser.add_argument("--attempts-out", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--request-timeout-seconds", type=int, default=1800)
    parser.add_argument("--pass-threshold", type=float, default=0.67)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--kinds", default="", help="Comma-separated row kinds to include.")
    parser.add_argument("--max-rows-per-kind", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.suite)
    rows = filter_rows(rows, parse_kinds(args.kinds), args.max_rows_per_kind)
    if args.sample_size:
        rows = sample_rows(rows, args.sample_size, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no rows selected for evaluation")
    if args.adaptive_token_budgets:
        budgets = parse_budgets(args.adaptive_token_budgets, args.max_new_tokens)
        summary = run_adaptive(args, rows, budgets)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["errors"] == 0 else 1

    budgets = parse_budgets(args.token_budgets, args.max_new_tokens)
    multi_budget = len(budgets) > 1

    summaries = []
    for budget in budgets:
        if budget < 1:
            raise ValueError("token budgets must be positive")
        out = output_for_budget(args.out, budget, multi_budget)
        summary_path = output_for_budget(args.summary, budget, multi_budget)
        summaries.append(run_budget(args, rows, budget, out, summary_path))

    if multi_budget:
        ladder_summary = {
            "ok": True,
            "suite": str(args.suite),
            "token_budgets": budgets,
            "summaries": summaries,
            "best_accuracy": max((item["accuracy"] for item in summaries), default=0.0),
            "best_avg_component_score": max((item["avg_component_score"] for item in summaries), default=0.0),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(ladder_summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(ladder_summary, indent=2, sort_keys=True))
        return 0 if all(item["errors"] == 0 for item in summaries) else 1

    print(json.dumps(summaries[0], indent=2, sort_keys=True))
    return 0 if summaries[0]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
