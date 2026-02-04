#!/usr/bin/env python3
import json, random
from pathlib import Path

INP = Path("data/router_train.jsonl")
OUT_TRAIN = Path("data/router_sft_train.jsonl")
OUT_VAL = Path("data/router_sft_val.jsonl")

random.seed(7)

def main():
    rows = [json.loads(l) for l in INP.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.shuffle(rows)

    n = len(rows)
    n_val = max(25, int(0.1 * n))  # 10% or at least 25
    val = rows[:n_val]
    train = rows[n_val:]

    OUT_TRAIN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    OUT_VAL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8")

    print(f"train={len(train)} -> {OUT_TRAIN}")
    print(f"val={len(val)} -> {OUT_VAL}")

if __name__ == "__main__":
    main()
