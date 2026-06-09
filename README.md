# discipline-patch

**Fix the behavior, not the knowledge.** This repo demonstrates a technique I call the **Discipline Patch**: when a fine-tuned LLM misbehaves, don't reach for more training data. Rewrite the model's own *failed evals* into corrective examples of the behavior you wanted, and retrain on just those.

The demonstration domain is security investigation: teaching an LLM to investigate like a disciplined SOC analyst over [MITRE ATT&CK®](https://attack.mitre.org/), via QLoRA fine-tuning, hard-negative refusal training, and a failure-driven eval and repair loop.

This is a research lab, not a product. The interesting problem here isn't teaching a model the ATT&CK framework — a base model already knows it. The hard part is teaching it to *behave* like a careful analyst: to map an observation to a technique and then say what evidence is still missing, to keep what the data shows separate from what it doesn't (no actor, attribution, or severity leaps), and to refuse a fabricated technique ID like `T9999.123` instead of inventing a confident answer. None of that is knowledge — it's behavior, and it's the whole reason a model like this is useful rather than just impressive. What the repo really demonstrates is a repeatable way to measure that behavior and improve it.

## A worked example: refusing a weak mapping

The prompt is deliberately a trap — an everyday network event that *looks* mappable but isn't:

> **Prompt:** A DHCP server shows a student laptop received an IP address at 8:02 AM. Can this alone be mapped to a MITRE ATT&CK technique?

A generic model tends to reach for a mapping anyway:

> This could map to Discovery — System Network Configuration Discovery — since the device obtained network configuration information.

Confident, plausible, and wrong. A DHCP lease is ordinary network activity; on its own it's nothing — and mapping it to an adversary technique is the overreach that floods a SOC queue with noise. After discipline tuning, the model holds the line:

> No — a DHCP lease by itself shouldn't be mapped to ATT&CK. It only shows a device joined or renewed on the network. A real mapping would need more: suspicious process execution, command-line activity, unusual DNS queries, authentication anomalies, signs of lateral movement, an endpoint alert, or a hit on known-malicious infrastructure. Treat this as context, not attack evidence.

Both versions know the framework. The difference is **behavior** — refusing the weak mapping, naming the missing evidence, and keeping observation separate from inference.

## What's here

```
data/
  build_attack_sft.py           # deterministic SFT rows from the public ATT&CK STIX corpus
                                #   (explainers, concise cards, mapping-boundary, mitigation,
                                #    procedure→technique, and fake-ID rejection hard-negatives)
  build_discipline_patch.py     # turn FAILED eval rows into targeted corrective training data
  run_training_row_arbiter.py   # deterministic data-quality gate over training rows
  mitre_probe_suite.v1.jsonl    # 12 analyst-discipline smoke probes (mapping, fake-ID rejection,
                                #   evidence boundaries, weak-evidence triage, defensive playbooks)
scripts/
  train_qwen_lora.py            # QLoRA / LoRA trainer (PEFT + HF Trainer)
  verify_train_gate.py          # fail-closed pre-train micro-gate
eval/
  run_sft_coverage_suite.py     # coverage eval engine + scoring (evidence, boundaries, refusal, format)
  score_mitre_answer.py         # score a single answer against the coverage rubric
  analyze_mitre_eval_failures.py# failure taxonomy over an eval run
```

## How it works

The whole thing is built on one idea: a model's own failures are the best training data you have, as long as you capture them cleanly. So it runs as a loop, not a one-shot job —

1. **Build** SFT data deterministically from the public MITRE ATT&CK enterprise STIX bundle — six row-types per technique, plus fake-ID rejection rows the model has to refuse.
2. **Gate** every row through a deterministic quality check (provenance, format, no fabricated IDs) before it's allowed near training.
3. **Train** a QLoRA adapter (4-bit, PEFT + HF `Trainer`).
4. **Evaluate for discipline, not recall** — is the evidence cited, the boundary held, the fake ID refused, the answer concise?
5. **Repair** — take whatever failed, turn it into targeted corrective examples, retrain, and run the same gate again to confirm the behavior actually moved.

The rule that makes it work — and the thing that surprised me most — is that a failed eval only becomes training data after it's rewritten into an explicit example of the behavior you wanted. You don't fix a misbehaving model with *more* data; you fix it with the *specific* data that targets the *specific* failure, captured from the model's own evals. If there's one idea worth taking from this repo, that's it.

## Results (v1 pilot — base model `Qwen3.6-27B`)

| Stage | Result |
|---|---|
| Base model | **0 / 12** smoke |
| Naive v1 adapter | 1 / 12 — *knew the mappings, failed analyst discipline* |
| After corrective **discipline patch** | **12 / 12** smoke (avg 0.969) → **71 / 71** on the v1 held-out *technique-explainer* eval split (avg 0.972) |

The corrective patch was 72 rows and about 18 minutes of retraining, and the part I found most telling is that the fixed answers got *shorter* and more structured, not longer — the discipline made the model more economical, not more verbose. Most passed at a 256-token budget.

### What this doesn't prove yet
I'll be straight about the edges: this is a validated pilot, not production. That 71/71 is specifically the v1 held-out *technique-explainer* split — broader coverage (procedure→technique disambiguation especially, plus concise cards, mitigation plans, and fake-ID rejection at scale) is still in progress, and procedure rows are the genuinely hard class. An early repetition-collapse failure (`T1590.003.003…`) got caught and patched with deterministic anti-repetition decoding. Adapters stay `experimental` until the full gates pass.

## Training setup

For the ML-minded, the adapter recipe (all defaults in `scripts/train_qwen_lora.py`, every one overridable):

| Knob | Value |
|---|---|
| Method | QLoRA: 4-bit nf4 base (double-quant, bf16 compute) + LoRA adapter via PEFT and HF `Trainer` |
| Rank / alpha / dropout | r=16, α=32, dropout 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (full attention + MLP) |
| Sequence length | 2048 |
| Batch | 1 × grad-accum 8 (effective 8), lr 2e-4, bf16, gradient checkpointing |
| Patch training | resumes the existing adapter (`--resume-adapter-dir`) and trains only on the corrective rows |
| DoRA | available behind `--use-dora` for like-for-like comparison against the same eval gates |

The base model here was `Qwen3.6-27B`, trained locally on a single GB10 (128 GB unified memory). Nothing is tied to that choice: the trainer takes any causal HF model, which is the point. The patch method is the portable part.

## Run it

```bash
pip install -r requirements.txt

# 1) build training data from the public ATT&CK STIX bundle, then gate it
python data/build_attack_sft.py        --help
python data/run_training_row_arbiter.py --help

# 2) train a QLoRA adapter
python scripts/train_qwen_lora.py      --help

# 3) evaluate analyst discipline
python eval/run_sft_coverage_suite.py  --help
```

*(Flags are intentionally left to each script's `--help` — the base model is swappable and the pipeline is model-agnostic.)*

## Data & provenance
Every training row is generated deterministically from the public MITRE ATT&CK® STIX corpus (`enterprise-attack`). There's no private, customer, or proprietary data anywhere in this repo — by design.

## License
[MIT](LICENSE). MITRE ATT&CK® is a registered trademark of The MITRE Corporation; this project is independent and not affiliated with or endorsed by MITRE.

---
More on the thinking behind disciplined, auditable AI: **[stoneytech.net](https://stoneytech.net)**.
