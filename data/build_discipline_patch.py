#!/usr/bin/env python3
"""Build corrective SFT rows from adaptive eval discipline failures."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are a defensive cyber analyst. For every ATT&CK mapping, separate "
    "observed evidence from inference. Always state what should not be inferred."
)
MAX_WRONG_IDS = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def assistant_answer(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def user_prompt(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return str(row.get("prompt") or "")


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
    match = re.search(rf"^{re.escape(label)}:\s*(.+?)\.", expected, flags=re.IGNORECASE | re.MULTILINE)
    return " ".join(match.group(1).split()) if match else ""


def detection_text(expected: str) -> str:
    match = re.search(r"Detection guidance:\s*(.+?)(?:\nRelated mitigations:|\nBoundary:|\nSource:|$)", expected, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return "Telemetry or logs must directly show the behavior being mapped, such as process, command-line, file, registry, authentication, network, or application events."


def attack_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:T\d{4}(?:\.\d{3})?|TA\d{4}|M\d{4})\b", text)))


def failure_diagnostics(failure: dict[str, Any]) -> dict[str, Any]:
    score = failure.get("score") if isinstance(failure.get("score"), dict) else {}
    diagnostics = score.get("diagnostics") if isinstance(score.get("diagnostics"), dict) else {}
    if diagnostics:
        return diagnostics
    repeated = sorted(set(re.findall(r"\bT\d{4}(?:\.\d{3}){2,}\b", str(failure.get("reply") or ""))))
    critical = set(score.get("critical_missing") or [])
    kind = str(failure.get("kind") or "")
    if repeated:
        failure_type = "repetition_collapse"
    elif kind == "fake_id_rejection":
        failure_type = "fake_id_acceptance"
    elif kind == "procedure_to_technique" and {"attack_id", "technique_name"} & critical:
        failure_type = "procedure_disambiguation"
    elif {"attack_id", "technique_name"} & critical:
        failure_type = "wrong_attack_mapping"
    elif {"evidence_reasoning", "boundary_reasoning"} & critical:
        failure_type = "analyst_discipline"
    else:
        failure_type = "unknown_failure"
    expected = str(failure.get("attack_id") or "")
    ids = attack_ids(str(failure.get("reply") or ""))
    return {
        "failure_type": failure_type,
        "detected_attack_ids": ids,
        "wrong_attack_ids": [item for item in ids if item != expected],
        "repeated_attack_fragments": repeated,
    }


def failure_type(failure: dict[str, Any]) -> str:
    return str(failure_diagnostics(failure).get("failure_type") or "unknown_failure")


def wrong_attack_ids(failure: dict[str, Any], expected_attack_id: str) -> list[str]:
    diagnostics = failure_diagnostics(failure)
    ids = diagnostics.get("wrong_attack_ids") or attack_ids(str(failure.get("reply") or ""))
    return [attack_id for attack_id in ids if attack_id != expected_attack_id][:MAX_WRONG_IDS]


def correction_answer(source_row: dict[str, Any], failure: dict[str, Any]) -> str:
    attack_id = str(source_row.get("attack_id") or "")
    kind = str(source_row.get("kind") or "")
    fail_type = failure_type(failure)
    if source_row.get("kind") == "fake_id_rejection":
        return "\n".join(
            [
                f"ATT&CK mapping: do not map {attack_id}.",
                f"Evidence required: verify {attack_id} against the versioned ATT&CK STIX corpus or official ATT&CK site before using it.",
                "Boundary / do not infer: do not invent a technique name, tactic, platform, detection, mitigation, actor, campaign, or incident severity for an unverified ATT&CK object.",
                "Confidence: use no confidence in the mapping until a valid official ATT&CK object is confirmed.",
            ]
        )
    expected = assistant_answer(source_row)
    name = expected_name(attack_id, expected) or "the referenced ATT&CK technique"
    tactics = expected_line(expected, "Tactics") or expected_line(expected, "Tactic") or "not listed in the provided object"
    platforms = expected_line(expected, "Platforms") or expected_line(expected, "Platform") or "not listed in the provided object"
    evidence = detection_text(expected)
    lines = [
        f"ATT&CK mapping: {attack_id} {name}.",
        (
            "Boundary / do not infer: this mapping does not by itself prove actor attribution, malware family, "
            "campaign, compromise scope, intent, business impact, or incident severity."
        ),
        f"Tactics: {tactics}.",
        f"Platforms: {platforms}.",
        f"Evidence required: {evidence}",
    ]
    if kind == "procedure_to_technique":
        lines.append(
            "Procedure decision: choose this technique only when the procedure evidence directly shows the named behavior; do not substitute a broader or sibling ATT&CK object."
        )
    if fail_type == "repetition_collapse":
        lines.append("Output discipline: write the ATT&CK ID once; do not repeat ID fragments or generate chained technique IDs.")
    elif fail_type == "analyst_discipline":
        lines.append("Analyst discipline: include both evidence required and boundary/do-not-infer language even when the mapping is correct.")
    wrong_ids = wrong_attack_ids(failure, attack_id)
    if wrong_ids and fail_type != "repetition_collapse":
        lines.append(
            f"Wrong-ID contrast: {', '.join(wrong_ids)} is not supported by this row unless the evidence directly shows those exact ATT&CK behaviors."
        )
    lines.append("Confidence: use low or medium confidence unless the observed telemetry directly matches the technique behavior.")
    return "\n".join(lines)


def make_row(source_name: str, source_row: dict[str, Any], prompt: str, answer: str, failure: dict[str, Any], variant: str) -> dict[str, Any]:
    diagnostics = failure_diagnostics(failure)
    fail_type = str(diagnostics.get("failure_type") or "unknown_failure")
    return {
        "schema": "stoneytech.spark_mitre_discipline_patch.v2",
        "source": source_name,
        "kind": f"discipline_correction_{fail_type}_{variant}",
        "attack_id": source_row.get("attack_id", failure.get("attack_id", "")),
        "object_id": source_row.get("object_id", ""),
        "source_kind": source_row.get("kind", failure.get("kind", "")),
        "failure_type": fail_type,
        "failure_missing": failure.get("score", {}).get("critical_missing", []),
        "wrong_attack_ids": diagnostics.get("wrong_attack_ids", []),
        "detected_attack_ids": diagnostics.get("detected_attack_ids", []),
        "repeated_attack_fragments": diagnostics.get("repeated_attack_fragments", []),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def build_rows(suite_rows: list[dict[str, Any]], failures: list[dict[str, Any]], source_name: str, repeat: int) -> list[dict[str, Any]]:
    by_attack_id = {str(row.get("attack_id") or ""): row for row in suite_rows if row.get("attack_id")}
    by_attack_id_kind = {
        (str(row.get("attack_id") or ""), str(row.get("kind") or "")): row for row in suite_rows if row.get("attack_id")
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for failure in failures:
        if failure.get("score", {}).get("passed"):
            continue
        attack_id = str(failure.get("attack_id") or "")
        fail_type = failure_type(failure)
        seen_key = (attack_id, str(failure.get("kind") or ""), fail_type)
        if not attack_id or seen_key in seen:
            continue
        seen.add(seen_key)
        source_row = by_attack_id_kind.get((attack_id, str(failure.get("kind") or ""))) or by_attack_id.get(attack_id)
        if not source_row:
            continue
        answer = correction_answer(source_row, failure)
        original_prompt = user_prompt(source_row)
        if fail_type == "procedure_disambiguation":
            prompts = [
                original_prompt,
                f"Procedure disambiguation drill: choose exactly one ATT&CK technique for {attack_id}, then state evidence required and boundary. Prompt: {original_prompt}",
                f"Correct the procedure mapping for {attack_id}. Do not choose a broader, sibling, or reconnaissance technique unless the procedure evidence supports it.",
            ]
        elif fail_type == "repetition_collapse":
            prompts = [
                original_prompt,
                f"Rewrite the ATT&CK answer for {attack_id} in 6 concise non-repeating lines. Write the technique ID once.",
                f"Correct the repeated-ID failure for {attack_id}. Include mapping, evidence required, and boundary without repeating ID fragments.",
            ]
        elif fail_type == "analyst_discipline":
            prompts = [
                original_prompt,
                f"Answer in 6 concise lines as a disciplined defensive analyst with explicit Evidence required and Boundary sections: {original_prompt}",
                f"Correct the weak ATT&CK answer for {attack_id}. Include evidence required and what must not be inferred. Keep it short.",
            ]
        else:
            prompts = [
                original_prompt,
                f"Answer in 6 concise lines as a disciplined defensive analyst with explicit Evidence required and Boundary sections: {original_prompt}",
                f"Correct the weak ATT&CK answer for {attack_id}. Include the mapping, evidence required, and what must not be inferred. Keep it short.",
            ]
        wrong_ids = wrong_attack_ids(failure, attack_id)
        if wrong_ids and fail_type not in {"repetition_collapse", "analyst_discipline"}:
            prompts.append(
                f"The previous answer selected {', '.join(wrong_ids)}. Select {attack_id} only if the procedure evidence supports it, and explain why the evidence boundary matters."
            )
        for _ in range(repeat):
            for index, prompt in enumerate(prompts, start=1):
                rows.append(make_row(source_name, source_row, prompt, answer, failure, f"v{index}"))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--adaptive-results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-train", type=Path)
    parser.add_argument("--out-combined", type=Path)
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()

    if args.repeat < 1 or args.repeat > 100:
        raise ValueError("--repeat must be between 1 and 100")
    source_name = args.adaptive_results.name
    patch_rows = build_rows(read_jsonl(args.suite), read_jsonl(args.adaptive_results), source_name, args.repeat)
    write_jsonl(args.out, patch_rows)
    combined_rows = []
    if args.base_train:
        combined_rows.extend(read_jsonl(args.base_train))
    combined_rows.extend(patch_rows)
    if args.out_combined:
        write_jsonl(args.out_combined, combined_rows)
    by_kind: dict[str, int] = {}
    for row in patch_rows:
        kind = str(row.get("kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(
        json.dumps(
            {
                "ok": True,
                "suite": str(args.suite),
                "adaptive_results": str(args.adaptive_results),
                "out": str(args.out),
                "patch_rows": len(patch_rows),
                "combined_rows": len(combined_rows) if args.out_combined else 0,
                "out_combined": str(args.out_combined or ""),
                "repeat": args.repeat,
                "by_kind": by_kind,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
