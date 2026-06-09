#!/usr/bin/env python3
"""Deterministic arbiter for MITRE training-row candidates.

The arbiter is intentionally local and conservative. It does not ask a model to
approve training data. It verifies row shape and ATT&CK discipline, applies a
small deterministic refiner for fixable writing defects, then splits rows into
accepted and rejected JSONL files with decision receipts.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any


ATTACK_TECHNIQUE_RE = re.compile(r"(?<![A-Z0-9])T\d{4}(?:\.\d{3})?(?![A-Z0-9.])", re.IGNORECASE)
ATTACK_OBJECT_RE = re.compile(r"(?<![A-Z0-9])(?:T\d{4}(?:\.\d{3})?|TA\d{4}|M\d{4})(?![A-Z0-9.])", re.IGNORECASE)
REPEATED_ATTACK_FRAGMENT_RE = re.compile(r"\bT\d{4}(?:\.\d{3}){2,}\b")
MAX_WRONG_IDS = 3
DEFAULT_MAX_ASSISTANT_WORDS = 260

STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

EVIDENCE_TERMS = (
    "evidence",
    "telemetry",
    "log",
    "logs",
    "observed",
    "behavior",
    "process",
    "command",
    "authentication",
    "network",
    "registry",
    "file",
    "event",
    "detection",
)

BOUNDARY_TERMS = (
    "boundary",
    "do not infer",
    "do not claim",
    "does not prove",
    "not prove",
    "not infer",
    "out of bounds",
    "from the technique id alone",
    "from this alone",
    "not because",
)

WRONG_ID_CONTRAST_TERMS = (
    "wrong-id contrast",
    "wrong id contrast",
    "do not use",
    "do not map",
    "not supported",
    "not substitute",
    "unless the evidence directly shows",
)

PROCEDURE_TERMS = (
    "procedure",
    "directly",
    "exact",
    "behavior",
    "broader",
    "sibling",
    "substitute",
    "evidence basis",
    "provided corpus",
)

REJECTION_TERMS = (
    "cannot verify",
    "not verify",
    "unverified",
    "do not map",
    "should not invent",
    "not a valid",
    "check the versioned",
    "official attack",
)


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_bundle(path: Path) -> list[dict[str, Any]]:
    parsed = json.loads(path.read_text())
    objects = parsed.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{path} is not an ATT&CK STIX bundle with objects[]")
    return [item for item in objects if isinstance(item, dict)]


def external_id(obj: dict[str, Any]) -> str:
    for ref in obj.get("external_references") or []:
        if isinstance(ref, dict) and ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return str(ref["external_id"])
    return ""


def ref_url(obj: dict[str, Any]) -> str:
    for ref in obj.get("external_references") or []:
        if isinstance(ref, dict) and ref.get("source_name") == "mitre-attack" and ref.get("url"):
            return str(ref["url"])
    return ""


def load_attack_catalog(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    catalog: dict[str, dict[str, str]] = {}
    for obj in read_bundle(path):
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        attack_id = external_id(obj)
        if not ATTACK_TECHNIQUE_RE.fullmatch(attack_id):
            continue
        catalog[attack_id] = {
            "attack_id": attack_id,
            "name": str(obj.get("name") or "").strip(),
            "object_id": str(obj.get("id") or ""),
            "url": ref_url(obj),
        }
    return catalog


def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("messages")
    return value if isinstance(value, list) else []


def message_content(row: dict[str, Any], role: str) -> str:
    for message in messages(row):
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def assistant_answer(row: dict[str, Any]) -> str:
    return message_content(row, "assistant")


def user_prompt(row: dict[str, Any]) -> str:
    return message_content(row, "user") or str(row.get("prompt") or "")


def set_assistant_answer(row: dict[str, Any], answer: str) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    updated_messages = messages(updated)
    for message in updated_messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = answer
            updated["messages"] = updated_messages
            return updated
    updated_messages.append({"role": "assistant", "content": answer})
    updated["messages"] = updated_messages
    return updated


def attack_objects(text: str) -> list[str]:
    return sorted({item.upper() for item in ATTACK_OBJECT_RE.findall(text)})


def attack_techniques(text: str) -> list[str]:
    return sorted({item.upper() for item in ATTACK_TECHNIQUE_RE.findall(text)})


def expected_name(attack_id: str, expected: str) -> str:
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


def expected_line(expected: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)(?:\.|\n|$)", expected, flags=re.IGNORECASE | re.MULTILINE)
    return " ".join(match.group(1).split()) if match else ""


def detection_text(expected: str) -> str:
    match = re.search(
        r"(?:Detection guidance|Useful evidence|Evidence basis|Evidence required):\s*(.+?)(?:\n[A-Z][A-Za-z /-]+:|$)",
        expected,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return " ".join(match.group(1).split())
    return (
        "telemetry or logs must directly show the behavior being mapped, such as process, command-line, "
        "file, registry, authentication, network, or application events."
    )


def word_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS]


def name_word_coverage(name: str, answer: str) -> float:
    name_words = set(word_tokens(name))
    if not name_words:
        return 1.0
    answer_words = set(word_tokens(answer))
    return len(name_words & answer_words) / len(name_words)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def is_fake_row(row: dict[str, Any]) -> bool:
    return "fake_id_rejection" in str(row.get("kind") or "")


def requires_strict_discipline(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind") or "")
    return (
        kind.startswith("discipline_correction")
        or "procedure" in kind
        or "mapping_boundary" in kind
        or "detection_plan" in kind
        or bool(row.get("failure_type"))
    )


def verifier(name: str, passed: bool, severity: str, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": passed, "severity": severity, "detail": detail}


def verify_row(
    row: dict[str, Any],
    catalog: dict[str, dict[str, str]],
    *,
    max_assistant_words: int = DEFAULT_MAX_ASSISTANT_WORDS,
) -> list[dict[str, Any]]:
    answer = assistant_answer(row)
    attack_id = str(row.get("attack_id") or "")
    kind = str(row.get("kind") or "")
    fake = is_fake_row(row)
    ids_in_answer = attack_objects(answer)
    wrong_technique_ids = [item for item in attack_techniques(answer) if item != attack_id]
    results: list[dict[str, Any]] = []

    role_contents = {role: message_content(row, role).strip() for role in ["system", "user", "assistant"]}
    messages_ok = all(role_contents.values())
    results.append(verifier("schema_messages", messages_ok, "critical", "requires non-empty system, user, and assistant messages"))

    if fake:
        format_ok = bool(ATTACK_OBJECT_RE.fullmatch(attack_id))
    else:
        format_ok = bool(ATTACK_TECHNIQUE_RE.fullmatch(attack_id))
    results.append(verifier("attack_id_format", format_ok, "critical", f"attack_id={attack_id!r}"))

    if not fake and catalog:
        official_ok = attack_id in catalog
        results.append(verifier("official_attack_id", official_ok, "critical", "attack_id must exist in the supplied ATT&CK catalog"))
    else:
        official_ok = True

    if fake:
        rejection_ok = contains_any(answer, REJECTION_TERMS)
        bad_claim = re.search(r"\bis\s+(?:a\s+)?(?:valid|real|official)\b", answer.lower()) and not rejection_ok
        results.append(
            verifier(
                "fake_id_rejection",
                bool(rejection_ok and not bad_claim),
                "critical",
                "invalid ATT&CK objects must be rejected without invented facts",
            )
        )
    else:
        mapping_ok = attack_id in ids_in_answer
        results.append(verifier("mapping_present", mapping_ok, "critical", "assistant answer must contain the expected ATT&CK ID"))
        if catalog and official_ok:
            official_name = catalog[attack_id]["name"]
            coverage = name_word_coverage(official_name, answer)
            results.append(
                verifier(
                    "official_name_match",
                    coverage >= 0.67,
                    "critical",
                    f"expected_name={official_name!r} coverage={coverage:.2f}",
                )
            )

    discipline_severity = "critical" if fake or requires_strict_discipline(row) else "warning"
    evidence_ok = contains_any(answer, EVIDENCE_TERMS)
    boundary_ok = contains_any(answer, BOUNDARY_TERMS)
    results.append(verifier("evidence_required", evidence_ok, discipline_severity, "answer should state the evidence or telemetry basis"))
    results.append(verifier("boundary_required", boundary_ok, discipline_severity, "answer should state what must not be inferred"))

    if "procedure" in kind or str(row.get("failure_type") or "") == "procedure_disambiguation":
        procedure_ok = contains_any(answer, PROCEDURE_TERMS)
        results.append(
            verifier(
                "procedure_grounding",
                procedure_ok,
                "critical",
                "procedure rows must tie mapping to the exact observed behavior and avoid sibling/broader substitutions",
            )
        )

    repeated = sorted(set(REPEATED_ATTACK_FRAGMENT_RE.findall(answer)))
    results.append(
        verifier(
            "no_repetition_collapse",
            not repeated,
            "critical",
            f"repeated_attack_fragments={repeated}",
        )
    )

    bounded_wrong_ids = len(wrong_technique_ids) <= MAX_WRONG_IDS
    results.append(
        verifier(
            "bounded_wrong_id_contrast",
            bounded_wrong_ids,
            "critical",
            f"wrong_attack_ids={wrong_technique_ids}",
        )
    )
    wrong_id_contrast_ok = not wrong_technique_ids or contains_any(answer, WRONG_ID_CONTRAST_TERMS)
    results.append(
        verifier(
            "wrong_id_contrast_adjudicated",
            wrong_id_contrast_ok,
            "critical",
            "extra technique IDs must be explicitly rejected or bounded as contrast, not mentioned as loose related mappings",
        )
    )

    answer_words = len(answer.split())
    results.append(
        verifier(
            "row_length",
            answer_words <= max_assistant_words,
            "warning",
            f"assistant_words={answer_words}, max={max_assistant_words}",
        )
    )
    return results


def critical_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in results if item["severity"] == "critical" and not item["passed"]]


def warning_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in results if item["severity"] == "warning" and not item["passed"]]


def non_refineable_failure_names(row: dict[str, Any], failures: list[dict[str, Any]]) -> set[str]:
    names = {item["name"] for item in failures}
    blocked = {"schema_messages", "attack_id_format", "official_attack_id"}
    if is_fake_row(row):
        blocked.discard("attack_id_format")
    return names & blocked


def fake_refined_answer(attack_id: str) -> str:
    return "\n".join(
        [
            f"ATT&CK mapping: do not map {attack_id}.",
            f"Evidence required: verify {attack_id} against the versioned ATT&CK STIX corpus or official ATT&CK site before using it.",
            "Boundary / do not infer: do not invent a technique name, tactic, platform, detection, mitigation, actor, campaign, compromise scope, or incident severity for an unverified ATT&CK object.",
            "Confidence: use no confidence in the mapping until a valid official ATT&CK object is confirmed.",
        ]
    )


def refined_answer(row: dict[str, Any], catalog: dict[str, dict[str, str]]) -> str:
    attack_id = str(row.get("attack_id") or "")
    if is_fake_row(row):
        return fake_refined_answer(attack_id)

    current = assistant_answer(row)
    name = ""
    if attack_id in catalog:
        name = catalog[attack_id]["name"]
    if not name:
        name = expected_name(attack_id, current)
    if not name:
        name = "the referenced ATT&CK technique"

    tactics = expected_line(current, "Tactics") or expected_line(current, "Tactic")
    platforms = expected_line(current, "Platforms") or expected_line(current, "Platform")
    evidence = detection_text(current)
    wrong_ids = [item for item in attack_techniques(current) if item != attack_id][:MAX_WRONG_IDS]

    lines = [f"ATT&CK mapping: {attack_id} {name}."]
    if tactics:
        lines.append(f"Tactics: {tactics}.")
    if platforms:
        lines.append(f"Platforms: {platforms}.")
    lines.extend(
        [
            f"Evidence required: {evidence}",
            (
                "Boundary / do not infer: this mapping does not by itself prove actor attribution, malware family, "
                "campaign, compromise scope, intent, business impact, or incident severity."
            ),
        ]
    )
    if "procedure" in str(row.get("kind") or "") or str(row.get("failure_type") or "") == "procedure_disambiguation":
        lines.append(
            "Procedure decision: choose this technique only when the procedure evidence directly shows the named behavior; do not substitute a broader or sibling ATT&CK object."
        )
    if wrong_ids:
        lines.append(
            f"Wrong-ID contrast: {', '.join(wrong_ids)} is not supported by this row unless the evidence directly shows those exact ATT&CK behaviors."
        )
    lines.append("Confidence: use low or medium confidence unless observed telemetry directly matches the technique behavior.")
    return "\n".join(lines)


def refine_row(row: dict[str, Any], catalog: dict[str, dict[str, str]]) -> dict[str, Any]:
    return set_assistant_answer(row, refined_answer(row, catalog))


def attach_metadata(
    row: dict[str, Any],
    *,
    status: str,
    history: list[dict[str, Any]],
    refinements: int,
) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    final_results = history[-1]["verifier_results"] if history else []
    failures = [item["name"] for item in critical_failures(final_results)]
    warnings = [item["name"] for item in warning_failures(final_results)]
    updated["arbiter_status"] = status
    updated["arbiter_attempts"] = len(history)
    updated["arbiter_refinements"] = refinements
    updated["arbiter_failures"] = failures
    updated["arbiter_warnings"] = warnings
    updated["arbiter_history"] = history
    return updated


def arbitrate_row(
    row: dict[str, Any],
    catalog: dict[str, dict[str, str]],
    *,
    max_refine_attempts: int = 2,
    max_assistant_words: int = DEFAULT_MAX_ASSISTANT_WORDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(row)
    history: list[dict[str, Any]] = []
    refinements = 0

    for attempt in range(max_refine_attempts + 1):
        results = verify_row(current, catalog, max_assistant_words=max_assistant_words)
        failures = critical_failures(results)
        history.append(
            {
                "attempt": attempt,
                "critical_failures": [item["name"] for item in failures],
                "warning_failures": [item["name"] for item in warning_failures(results)],
                "verifier_results": results,
            }
        )
        if not failures:
            final_row = attach_metadata(current, status="accepted", history=history, refinements=refinements)
            decision = decision_receipt(final_row, status="accepted", history=history)
            return final_row, decision
        blocked = non_refineable_failure_names(current, failures)
        if blocked or attempt >= max_refine_attempts:
            final_row = attach_metadata(current, status="rejected", history=history, refinements=refinements)
            decision = decision_receipt(final_row, status="rejected", history=history)
            if blocked:
                decision["non_refineable_failures"] = sorted(blocked)
            return final_row, decision
        current = refine_row(current, catalog)
        refinements += 1

    final_row = attach_metadata(current, status="rejected", history=history, refinements=refinements)
    return final_row, decision_receipt(final_row, status="rejected", history=history)


def decision_receipt(row: dict[str, Any], *, status: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    final_results = history[-1]["verifier_results"] if history else []
    return {
        "status": status,
        "attack_id": str(row.get("attack_id") or ""),
        "kind": str(row.get("kind") or ""),
        "source_kind": str(row.get("source_kind") or ""),
        "failure_type": str(row.get("failure_type") or ""),
        "attempts": len(history),
        "refinements": int(row.get("arbiter_refinements") or 0),
        "critical_failures": [item["name"] for item in critical_failures(final_results)],
        "warning_failures": [item["name"] for item in warning_failures(final_results)],
        "history": history,
    }


def increment(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def summarize(
    *,
    candidates: Path,
    accepted_out: Path,
    rejected_out: Path,
    decision_out: Path,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    catalog: dict[str, dict[str, str]],
) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_failure_type: dict[str, int] = {}
    by_verifier_failure: dict[str, int] = {}
    refined_rows = 0
    for decision in decisions:
        increment(by_status, str(decision.get("status") or "unknown"))
        increment(by_kind, str(decision.get("kind") or "unknown"))
        if decision.get("failure_type"):
            increment(by_failure_type, str(decision["failure_type"]))
        if int(decision.get("refinements") or 0) > 0:
            refined_rows += 1
        for name in decision.get("critical_failures") or []:
            increment(by_verifier_failure, str(name))
    total = len(decisions)
    return {
        "ok": len(rejected) == 0,
        "usable": len(accepted) > 0,
        "schema": "stoneytech.spark_training_row_arbiter.summary.v1",
        "generated_at": utc_stamp(),
        "candidates": str(candidates),
        "accepted_out": str(accepted_out),
        "rejected_out": str(rejected_out),
        "decision_out": str(decision_out),
        "candidate_rows": total,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "refined_rows": refined_rows,
        "catalog_techniques": len(catalog),
        "by_status": by_status,
        "by_kind": by_kind,
        "by_failure_type": by_failure_type,
        "by_verifier_failure": by_verifier_failure,
    }


def default_output_paths(candidates: Path, label: str) -> tuple[Path, Path, Path, Path]:
    base_dir = candidates.parent
    safe_label = label or candidates.stem
    return (
        base_dir / f"{safe_label}-arbiter-accepted.jsonl",
        base_dir / f"{safe_label}-arbiter-rejected.jsonl",
        base_dir / f"{safe_label}-arbiter-decisions.jsonl",
        base_dir / f"{safe_label}-arbiter-summary.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--accepted-out", type=Path)
    parser.add_argument("--rejected-out", type=Path)
    parser.add_argument("--decision-out", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--enterprise-attack-json", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--max-refine-attempts", type=int, default=2)
    parser.add_argument("--max-assistant-words", type=int, default=DEFAULT_MAX_ASSISTANT_WORDS)
    args = parser.parse_args()

    if args.max_refine_attempts < 0 or args.max_refine_attempts > 10:
        raise ValueError("--max-refine-attempts must be between 0 and 10")
    if args.max_assistant_words < 60 or args.max_assistant_words > 2000:
        raise ValueError("--max-assistant-words must be between 60 and 2000")

    accepted_default, rejected_default, decision_default, summary_default = default_output_paths(
        args.candidates, args.label or args.candidates.stem
    )
    accepted_out = args.accepted_out or accepted_default
    rejected_out = args.rejected_out or rejected_default
    decision_out = args.decision_out or decision_default
    summary_path = args.summary or summary_default

    catalog = load_attack_catalog(args.enterprise_attack_json)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for index, row in enumerate(read_jsonl(args.candidates), start=1):
        final_row, decision = arbitrate_row(
            row,
            catalog,
            max_refine_attempts=args.max_refine_attempts,
            max_assistant_words=args.max_assistant_words,
        )
        decision["index"] = index
        decisions.append(decision)
        if decision["status"] == "accepted":
            accepted.append(final_row)
        else:
            rejected.append(final_row)

    write_jsonl(accepted_out, accepted)
    write_jsonl(rejected_out, rejected)
    write_jsonl(decision_out, decisions)
    summary = summarize(
        candidates=args.candidates,
        accepted_out=accepted_out,
        rejected_out=rejected_out,
        decision_out=decision_out,
        accepted=accepted,
        rejected=rejected,
        decisions=decisions,
        catalog=catalog,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
