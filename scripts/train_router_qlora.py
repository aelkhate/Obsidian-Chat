#!/usr/bin/env python3
"""
scripts/train_router_qlora.py  (WINDOWS/WDDM + RTX 4060 Ti 16GB)

Fixes:
- Avoid Windows VRAM thrash: seq=256, bs=1
- Avoid GradScaler BF16 crash:
  Use bf16=True, fp16=False (no grad scaling)
- Force GPU placement at load time
- Align special token IDs
- LoRA attention-only (enough for router, less VRAM)
"""

import json
import os
import inspect
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TRAIN_PATH = "data/router_sft_train.jsonl"
VAL_PATH = "data/router_sft_val.jsonl"
OUT_DIR = "data/lora/router_qwen25_7b_wddm_fit"

MAX_SEQ_LEN = 256
MAX_STEPS = 120

PER_DEVICE_BS = 1
GRAD_ACCUM = 8  # effective batch = 8

LR = 1e-4
WARMUP_STEPS = 10

LOGGING_STEPS = 10
SAVE_STEPS = 60
EVAL_STEPS = 60

DATALOADER_WORKERS = 0
ENABLE_PACKING = True


def build_training_args(out_dir: str, has_val: bool) -> TrainingArguments:
    common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to=[],
        dataloader_num_workers=DATALOADER_WORKERS,
        optim="paged_adamw_8bit",
        max_grad_norm=0.0,  # IMPORTANT: avoid unscale+clip path issues
    )

    # Force BF16 (no GradScaler). This avoids your exact crash.
    if has_val:
        try:
            return TrainingArguments(
                **common,
                bf16=True,
                fp16=False,
                evaluation_strategy="steps",
                eval_steps=EVAL_STEPS,
            )
        except TypeError:
            pass

    return TrainingArguments(**common, bf16=True, fp16=False)


def main():
    if not os.path.exists("src"):
        raise SystemExit("Run from repo root: python -m scripts.train_router_qlora")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from src.router_agent import ROUTER_SYSTEM, FEWSHOT

    if not os.path.exists(TRAIN_PATH):
        raise SystemExit(f"Missing {TRAIN_PATH}. Run: python -m scripts.split_router_dataset")

    has_val = os.path.exists(VAL_PATH)
    ds_train = load_dataset("json", data_files=TRAIN_PATH, split="train")
    ds_val = load_dataset("json", data_files=VAL_PATH, split="train") if has_val else None

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # BF16 compute dtype for 4-bit matmuls
    compute_dtype = torch.bfloat16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Your Transformers deprecates torch_dtype -> use dtype=
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        dtype=compute_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Align token ids (BOS may be None; that's fine)
    model.config.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id

    try:
        model.config.use_cache = False
    except Exception:
        pass

    # LoRA (attention-only) to reduce VRAM
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    # Keep speed; if you still OOM, enable checkpointing instead
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    FEWSHOT_PREFIX = "\n\n".join(
        f"Example Question: {ex['q']}\nExample Plan:\n{json.dumps(ex['plan'], ensure_ascii=False)}"
        for ex in FEWSHOT
    )

    def router_user_prompt(user_q: str) -> str:
        return f"{FEWSHOT_PREFIX}\n\nUser Question: {user_q}\nReturn ONLY the JSON plan."

    def to_text(ex):
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": router_user_prompt(ex["user_q"])},
            {"role": "assistant", "content": ex["target_json"]},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds_train2 = ds_train.map(to_text, remove_columns=ds_train.column_names)
    ds_val2 = ds_val.map(to_text, remove_columns=ds_val.column_names) if ds_val is not None else None

    print("GPU:", torch.cuda.get_device_name(0))
    print("Sample text chars:", len(ds_train2[0]["text"]))
    print("pad_token_id:", tok.pad_token_id, "eos_token_id:", tok.eos_token_id, "bos_token_id:", tok.bos_token_id)

    args = build_training_args(OUT_DIR, has_val=(ds_val2 is not None))

    sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {"model": model, "train_dataset": ds_train2, "args": args}
    if ds_val2 is not None:
        trainer_kwargs["eval_dataset"] = ds_val2

    if "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tok
    elif "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tok

    if "dataset_text_field" in sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LEN
    elif "max_length" in sig.parameters:
        trainer_kwargs["max_length"] = MAX_SEQ_LEN

    if ENABLE_PACKING and "packing" in sig.parameters:
        trainer_kwargs["packing"] = True

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    trainer.model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Saved router LoRA adapter -> {OUT_DIR}")


if __name__ == "__main__":
    main()
