# attack-analyst-lora

Teaching an LLM to **investigate like a disciplined SOC analyst** over [MITRE ATT&CK®](https://attack.mitre.org/) — via QLoRA fine-tuning, hard-negative refusal training, and a failure-driven eval → repair loop.

This is a research lab, not a product. It fine-tunes an open model to:

- map observations to **ATT&CK technique IDs**, and call out the **evidence still needed**;
- **separate evidence from inference** — no actor / attribution / severity leaps;
- **refuse fabricated technique IDs** (e.g. `T9999.123`);
- answer in a concise, evaluable analyst format.

The point isn't fact recall — base models can already recite ATT&CK. The point is **analyst discipline**, and a **repeatable process** to measure and improve it.

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

## The method — a closed improvement loop

1. **Build** SFT data deterministically from the **public MITRE ATT&CK enterprise STIX bundle** — six row-types per technique, plus **fake-ID rejection** rows (hard negatives the model must refuse).
2. **Gate** every training row through a deterministic arbiter (provenance, format, no fabricated IDs).
3. **Train** a QLoRA adapter (4-bit, PEFT + HF `Trainer`).
4. **Evaluate for discipline, not recall** — evidence cited, boundaries respected, fake IDs refused, concise format.
5. **Repair** — convert the *failed* eval rows into a targeted "discipline patch" dataset, retrain, and re-run the **same gate** to confirm the behavior actually improved.

The core idea: **failed evals become training data only after they're converted into explicit analyst-discipline examples.** Model improvement is treated as an engineering loop, not a one-time training event.

## Results (v1 pilot — base model `Qwen3.6-27B`)

| Stage | Result |
|---|---|
| Base model | **0 / 12** smoke |
| Naive v1 adapter | 1 / 12 — *knew the mappings, failed analyst discipline* |
| After corrective **discipline patch** | **12 / 12** smoke (avg 0.969) → **71 / 71** on the v1 held-out *technique-explainer* eval split (avg 0.972) |

The corrective patch was 72 rows and ~18 minutes of retraining. It made answers shorter, more structured, and reliable — most passed at a **256-token** budget.

### Honest limits
This is a **validated pilot, not production.** The 71/71 figure is specifically the v1 held-out *technique-explainer* split. Expanded v2 coverage (procedure→technique disambiguation, concise cards, mitigation plans, fake-ID rejection at scale) is **still in progress** — procedure rows are the hard class, and an early repetition-collapse failure (`T1590.003.003…`) was caught and mitigated with deterministic anti-repetition decoding (`repetition_penalty`, `no_repeat_ngram_size`). Adapters stay `experimental` / `candidate` until full eval gates pass.

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
All training data is generated **deterministically from the public MITRE ATT&CK® STIX corpus** (`enterprise-attack`). **No private, customer, or proprietary data is used anywhere in this repository.**

## License
[MIT](LICENSE). MITRE ATT&CK® is a registered trademark of The MITRE Corporation; this project is independent and not affiliated with or endorsed by MITRE.

---
More on the thinking behind disciplined, auditable AI: **[stoneytech.net](https://stoneytech.net)**.
