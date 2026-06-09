#!/usr/bin/env python3
"""Build small SFT rows from MITRE ATT&CK STIX JSON.

Input is the official enterprise-attack STIX bundle JSON. This script does not
download data; keep source acquisition explicit so every corpus has a versioned
file pointer and date.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are a defensive cyber analyst. Use MITRE ATT&CK facts only when the "
    "evidence supports them. Prefer technique IDs, name missing evidence, and "
    "do not invent ATT&CK objects."
)


def read_bundle(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError("STIX bundle must contain an objects list")
    return [item for item in objects if isinstance(item, dict)]


def external_id(obj: dict[str, Any]) -> str:
    for ref in obj.get("external_references") or []:
        if not isinstance(ref, dict):
            continue
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return str(ref["external_id"])
    return ""


def ref_url(obj: dict[str, Any]) -> str:
    for ref in obj.get("external_references") or []:
        if not isinstance(ref, dict):
            continue
        if ref.get("source_name") == "mitre-attack" and ref.get("url"):
            return str(ref["url"])
    return ""


def short_text(value: Any, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def relationship_index(objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        target = obj.get("target_ref")
        if isinstance(target, str):
            index.setdefault(target, []).append(obj)
    return index


def make_row(
    source_name: str,
    kind: str,
    attack_id: str,
    object_id: Any,
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    return {
        "schema": "stoneytech.spark_mitre_sft.v1",
        "source": source_name,
        "kind": kind,
        "attack_id": attack_id,
        "object_id": object_id,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def active_techniques(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        obj
        for obj in objects
        if obj.get("type") == "attack-pattern" and external_id(obj) and not obj.get("revoked")
    ]


def technique_rows(objects: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = {obj.get("id"): obj for obj in objects if isinstance(obj.get("id"), str)}
    rels = relationship_index(objects)

    for tech in active_techniques(objects):
        tid = external_id(tech)
        name = str(tech.get("name") or "").strip()
        tactics = [
            phase.get("phase_name", "").replace("-", " ")
            for phase in tech.get("kill_chain_phases") or []
            if isinstance(phase, dict) and phase.get("phase_name")
        ]
        platforms = ", ".join(tech.get("x_mitre_platforms") or [])
        detection = short_text(tech.get("x_mitre_detection"), 700)
        mitigations = []
        for rel in rels.get(str(tech.get("id")), []):
            if rel.get("relationship_type") != "mitigates":
                continue
            source = by_id.get(rel.get("source_ref"))
            if source and source.get("type") == "course-of-action":
                mid = external_id(source)
                label = source.get("name")
                if mid or label:
                    mitigations.append(f"{mid} {label}".strip())

        prompt = (
            f"Teach a junior defensive analyst the ATT&CK entry {tid} {name}. "
            "Include what evidence supports mapping it and what should not be inferred."
        )
        answer_parts = [
            f"{tid} is {name}.",
            f"Tactics: {', '.join(tactics) if tactics else 'not listed in the provided object'}.",
            f"Platforms: {platforms or 'not listed in the provided object'}.",
            f"Description: {short_text(tech.get('description'), 800)}",
        ]
        if detection:
            answer_parts.append(f"Detection guidance: {detection}")
        if mitigations:
            answer_parts.append(f"Related mitigations: {', '.join(mitigations[:6])}.")
        answer_parts.append(
            "Boundary: do not claim attribution, malware family, or incident severity from the technique ID alone."
        )
        if ref_url(tech):
            answer_parts.append(f"Source: {ref_url(tech)}")
        rows.append(make_row(source_name, "technique_explainer", tid, tech.get("id"), prompt, "\n".join(answer_parts)))

        rows.append(
            make_row(
                source_name,
                "technique_mapping_boundary",
                tid,
                tech.get("id"),
                (
                    f"An alert may involve {name}. What ATT&CK mapping is allowed, "
                    "what evidence is needed, and what claims stay out of bounds?"
                ),
                "\n".join(
                    [
                        f"Allowed mapping: {tid} {name}, only when the observed behavior matches the technique.",
                        f"Tactics: {', '.join(tactics) if tactics else 'not listed in the provided object'}.",
                        f"Useful evidence: {detection or 'process, authentication, network, file, registry, or application logs that directly show the behavior.'}",
                        "Out of bounds: do not claim actor identity, malware family, campaign, business impact, or confirmed compromise from the ATT&CK mapping alone.",
                    ]
                ),
            )
        )

        rows.append(
            make_row(
                source_name,
                "technique_concise_card",
                tid,
                tech.get("id"),
                f"Return a concise analyst card for {tid} {name}.",
                "\n".join(
                    [
                        f"Technique: {tid} {name}.",
                        f"Tactic: {', '.join(tactics) if tactics else 'not listed'}.",
                        f"Platform: {platforms or 'not listed'}.",
                        "Confidence rule: map it only when the evidence shows the behavior, not because the object name sounds similar.",
                    ]
                ),
            )
        )

        if detection:
            rows.append(
                make_row(
                    source_name,
                    "technique_detection_plan",
                    tid,
                    tech.get("id"),
                    f"Build a defensive detection and triage plan for ATT&CK {tid} {name}.",
                    "\n".join(
                        [
                            f"ATT&CK mapping: {tid} {name}.",
                            f"Detection guidance from the corpus: {detection}",
                            "Triage: collect the logs that prove the behavior, compare timing and host/account context, then separate confirmed facts from hypotheses.",
                            "Boundary: absence of one log source does not prove absence of the behavior, and the technique does not prove attribution.",
                        ]
                    ),
                )
            )

        if mitigations:
            rows.append(
                make_row(
                    source_name,
                    "technique_mitigation_plan",
                    tid,
                    tech.get("id"),
                    f"What mitigations from ATT&CK relate to {tid} {name}, and how should a small IT team use them?",
                    "\n".join(
                        [
                            f"Technique: {tid} {name}.",
                            f"Related mitigations: {', '.join(mitigations[:8])}.",
                            "Use mitigations as risk-reduction controls, then verify with telemetry. A mitigation does not prove the technique occurred.",
                        ]
                    ),
                )
            )
    return rows


def procedure_rows(objects: list[dict[str, Any]], source_name: str, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    by_id = {obj.get("id"): obj for obj in objects if isinstance(obj.get("id"), str)}
    technique_ids = {str(obj.get("id")) for obj in active_techniques(objects)}
    for rel in objects:
        if len(rows) >= limit:
            break
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "uses":
            continue
        target_ref = rel.get("target_ref")
        if target_ref not in technique_ids:
            continue
        source = by_id.get(rel.get("source_ref"))
        target = by_id.get(target_ref)
        if not source or not target:
            continue
        source_name_label = str(source.get("name") or "").strip()
        target_id = external_id(target)
        target_name = str(target.get("name") or "").strip()
        if not source_name_label or not target_id or not target_name:
            continue
        rel_desc = short_text(rel.get("description"), 600)
        prompt = (
            f"Procedure evidence mentions {source_name_label}: {rel_desc or target_name}. "
            "Map the ATT&CK technique and name the evidence boundary."
        )
        answer = "\n".join(
            [
                f"Best supported ATT&CK mapping from the provided corpus: {target_id} {target_name}.",
                f"Procedure source: {source_name_label}.",
                f"Evidence basis: {rel_desc or 'the corpus relationship states this source uses the technique.'}",
                "Boundary: this maps the described procedure to a technique; it does not prove the same source, actor, or malware is present in a new environment.",
            ]
        )
        rows.append(make_row(source_name, "procedure_to_technique", target_id, target.get("id"), prompt, answer))
    return rows


def fake_id_rows(source_name: str) -> list[dict[str, Any]]:
    fake_ids = ["T9999", "T0000.999", "TA9999", "M9999", "T1059.999"]
    rows = []
    for fake_id in fake_ids:
        answer = (
            f"I cannot verify {fake_id} as a valid MITRE ATT&CK object from the provided corpus. "
            "I should not invent a name, tactic, detection, or mitigation for it. The next step is to "
            "check the versioned ATT&CK STIX corpus or official ATT&CK site, then answer from a valid object."
        )
        rows.append(make_row(source_name, "fake_id_rejection", fake_id, fake_id, f"Explain MITRE ATT&CK object {fake_id} and give detections.", answer))
        rows.append(make_row(source_name, "fake_id_rejection", fake_id, fake_id, f"Is {fake_id} a real ATT&CK technique? Keep the answer defensive and factual.", answer))
        rows.append(make_row(source_name, "fake_id_rejection", fake_id, fake_id, f"An analyst wrote {fake_id} in a ticket. What should you do before using that mapping?", answer))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enterprise-attack-json", required=True, type=Path)
    parser.add_argument("--out-train", default=Path("artifacts/data/mitre_train.jsonl"), type=Path)
    parser.add_argument("--out-eval", default=Path("artifacts/data/mitre_eval.jsonl"), type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--procedure-limit", type=int, default=2500)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_name = args.enterprise_attack_json.name
    objects = read_bundle(args.enterprise_attack_json)
    rows = technique_rows(objects, source_name)
    rows.extend(procedure_rows(objects, source_name, args.procedure_limit))
    rows.extend(fake_id_rows(source_name))
    if args.limit:
        rows = rows[: args.limit]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_count = max(1, int(len(rows) * args.eval_ratio)) if rows else 0
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_eval, eval_rows)
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[str(row.get("kind") or "unknown")] = by_kind.get(str(row.get("kind") or "unknown"), 0) + 1
    print(
        json.dumps(
            {
                "ok": True,
                "source": source_name,
                "total_rows": len(rows),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "by_kind": by_kind,
                "out_train": str(args.out_train),
                "out_eval": str(args.out_eval),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
