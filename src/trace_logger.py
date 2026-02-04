# src/trace_logger.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """
    Append one JSON object per line (JSONL).
    Creates parent directory if needed.
    """
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def now_ts() -> float:
    return time.time()


def make_run_id() -> str:
    # stable-ish id without extra deps
    return str(int(time.time() * 1000))


def safe_str(x: Any, max_len: int = 5000) -> str:
    s = "" if x is None else str(x)
    if len(s) > max_len:
        return s[:max_len] + "...[TRUNCATED]"
    return s


def build_trace(
    *,
    run_id: str,
    user_q: str,
    intent: str,
    plan: Optional[Dict[str, Any]],
    answer: str,
    sources: Any,
    verifier: Optional[Dict[str, Any]],
    repaired: bool = False,
    repaired_answer: Optional[str] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "ts": now_ts(),
        "user_q": safe_str(user_q, 5000),
        "intent": intent,
        "plan": plan,
        "answer": safe_str(answer, 15000),
        "sources": sources,
        "verifier": verifier,
        "repaired": bool(repaired),
        "repaired_answer": safe_str(repaired_answer, 15000) if repaired_answer else None,
        "notes": notes or {},
    }
