#!/usr/bin/env python3
"""Train a Qwen LoRA adapter on MITRE SFT rows."""
from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_rows(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def load_processor(model_id: str, model_class: str) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    if model_class == "causal-lm":
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return processor
    except Exception:
        if model_class == "image-text-to-text":
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


def tokenizer_from_processor(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in messages
            if isinstance(item, dict) and item.get("role") and item.get("content")
        ]
    prompt = str(row.get("prompt") or "").strip()
    completion = str(row.get("completion") or row.get("answer") or "").strip()
    if not prompt or not completion:
        raise ValueError("row must contain messages or prompt+completion")
    return [
        {"role": "system", "content": "You are a defensive cyber analyst."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


class SftRows:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples = []
        for row in rows:
            try:
                text = tokenizer.apply_chat_template(row_messages(row), tokenize=False, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(row_messages(row), tokenize=False)
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
            labels = list(encoded["input_ids"])
            self.examples.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--model-class", choices=["auto", "causal-lm", "image-text-to-text"], default="auto")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--eval-jsonl", type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts/adapters/qwen-mitre-lora"), type=Path)
    parser.add_argument("--resume-adapter-dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--trainer-eval-strategy", choices=["no", "steps", "epoch"], default="no")
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-dora", action="store_true")
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated PEFT target modules.",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.time()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, Trainer, TrainingArguments

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = load_processor(args.model, args.model_class)
    tokenizer = tokenizer_from_processor(processor)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if args.model_class == "image-text-to-text":
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    elif args.model_class == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.resume_adapter_dir:
        if args.use_dora:
            raise RuntimeError("--use-dora cannot be added while resuming an existing non-DoRA adapter")
        model = PeftModel.from_pretrained(model, str(args.resume_adapter_dir), is_trainable=True)
    else:
        lora_kwargs = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        }
        if args.use_dora:
            if "use_dora" not in inspect.signature(LoraConfig.__init__).parameters:
                raise RuntimeError("Installed PEFT version does not support --use-dora")
            lora_kwargs["use_dora"] = True
        peft_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_rows = read_rows(args.train_jsonl, args.max_rows)
    eval_rows = (
        read_rows(args.eval_jsonl, max(1, int(args.max_rows * 0.1)) if args.max_rows else 0)
        if args.eval_jsonl and args.trainer_eval_strategy != "no"
        else []
    )
    train_ds = SftRows(train_rows, tokenizer, args.max_length)
    eval_ds = SftRows(eval_rows, tokenizer, args.max_length) if eval_rows else None
    save_strategy = args.save_strategy
    save_steps = args.save_steps
    if save_strategy == "steps" and save_steps <= 0:
        raise ValueError("--save-steps must be positive when save strategy is steps")

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "bf16": args.bf16,
        "logging_steps": 5,
        "save_strategy": save_strategy,
        "save_total_limit": args.save_total_limit,
        "report_to": [],
        "remove_unused_columns": False,
    }
    if save_strategy == "steps":
        training_kwargs["save_steps"] = save_steps
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = args.trainer_eval_strategy if eval_ds else "no"
    if eval_ds and args.trainer_eval_strategy == "steps":
        if args.eval_steps <= 0:
            raise ValueError("--eval-steps must be positive when trainer eval strategy is steps")
        training_kwargs["eval_steps"] = args.eval_steps
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollator(tokenizer),
    )
    result = trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    receipt = {
        "ok": True,
        "model": args.model,
        "model_class": args.model_class,
        "adapter_dir": str(args.output_dir),
        "resume_adapter_dir": str(args.resume_adapter_dir or ""),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "use_dora": args.use_dora,
        "load_in_4bit": args.load_in_4bit,
        "max_steps": args.max_steps,
        "save_strategy": save_strategy,
        "save_steps": save_steps if save_strategy == "steps" else 0,
        "save_total_limit": args.save_total_limit,
        "trainer_eval_strategy": args.trainer_eval_strategy if eval_ds else "no",
        "eval_steps": args.eval_steps if eval_ds and args.trainer_eval_strategy == "steps" else 0,
        "gradient_checkpointing": args.gradient_checkpointing,
        "target_modules": [item.strip() for item in args.target_modules.split(",") if item.strip()],
        "train_metrics": result.metrics,
        "seconds": round(time.time() - started, 3),
    }
    (args.output_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
