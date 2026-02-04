# src/verifier.py
"""
Deterministic Verifier for Obsidian Chat

Purpose:
- Enforce citation discipline + structural guarantees deterministically.
- No LLM usage. Pure string/regex rules.

Contract:
verify_answer(answer_text, sources, intent) -> dict
- Returns {"status":"PASS"} OR {"status":"FAIL", "reason":..., "details":...}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional


Intent = Literal["RAG_QA", "DOC_EXTRACT", "BROWSE", "SYNTHESIS"]

REFUSAL_TEXT = "I couldn't find that in your notes."

# Allow trailing punctuation after the citation, e.g. "(source: [1])."
# Also allow optional closing quotes/brackets common in markdown or prose.
# Examples matched:
#   "... (source: [1])"
#   "... (source: [1])."
#   "... (source: [1])!"
#   "... (source: [1])”)."
CITATION_AT_END_RE = re.compile(
    r"\(source:\s*\[(\d+)\]\)\s*[\.\!\?\)\]\}\"'”’]*\s*$",
    re.IGNORECASE,
)

# Sentence-ish splitter (simple + deterministic)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _pass() -> Dict[str, Any]:
    return {"status": "PASS"}


def _fail(reason: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"status": "FAIL", "reason": reason, "details": details or {}}


def _allowed_citation_ids(num_sources: int) -> set[str]:
    return {str(i) for i in range(1, num_sources + 1)}


def _is_exempt_line(line: str) -> bool:
    """
    Lines that are not factual claims and should not require citations.
    This is intentionally conservative (small allowlist).
    """
    s = line.strip()
    if not s:
        return True

    # Markdown headings / separators / code fences / blockquotes
    if s.startswith("#"):
        return True
    if s in ("---", "***", "___"):
        return True
    if s.startswith("```"):
        return True
    if s.startswith(">"):
        return True

    # Pure bullets without content (rare)
    if s in ("-", "*"):
        return True

    return False


def _iter_claim_units(answer_text: str) -> List[str]:
    """
    Returns a list of "claim units" that MUST be cited.

    Strategy:
    1) Split by lines.
    2) For each non-exempt line:
       - If it's a bullet line, treat the entire line as a claim unit.
       - Else split into sentences and treat each sentence as a claim unit.
    """
    units: List[str] = []
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if _is_exempt_line(line):
            continue

        # Bullet lines are usually standalone factual claims
        if line.startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ")):
            units.append(line)
            continue

        parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(line) if p.strip()]
        if not parts:
            units.append(line)
        else:
            units.extend(parts)

    return units


def verify_answer(answer_text: str, sources: List[Dict[str, Any]], intent: Intent) -> Dict[str, Any]:
    """
    Deterministically verify answer_text against sources + intent.

    Rules enforced:
    1) Valid citation IDs only: [1..N]
    2) Every claim unit must end with (source: [n]) (optionally followed by punctuation)
    3) For BROWSE and DOC_EXTRACT, require exactly 1 source (your current design)
    4) Refusal consistency:
       - If refusal text appears: sources must be empty and no citations should exist
    """
    text = (answer_text or "").strip()
    if not text:
        return _fail("Empty answer")

    # Refusal path
    if REFUSAL_TEXT in text:
        if sources:
            return _fail("Refusal with non-empty sources", {"sources_len": len(sources)})
        # Ensure refusal isn't decorated with citations
        if "(source:" in text.lower():
            return _fail("Refusal contains citations", {})
        return _pass()

    # Intent-specific single-source constraint (per your current orchestrator behavior)
    if intent in ("DOC_EXTRACT", "BROWSE") and len(sources) != 1:
        return _fail(f"{intent} must have exactly one source", {"sources_len": len(sources)})

    # If there are no sources, the only allowed answer is refusal
    if not sources:
        return _fail("Non-refusal answer with empty sources", {})

    allowed = _allowed_citation_ids(len(sources))
    claim_units = _iter_claim_units(text)

    if not claim_units:
        return _fail("Answer contains no verifiable claim units", {})

    for unit in claim_units:
        unit_stripped = (unit or "").strip()

        # Exempt headings and structural lead-ins
        if not unit_stripped:
            continue
        if unit_stripped.startswith("#"):
            continue
        if unit_stripped.endswith(":"):
            continue

        m = CITATION_AT_END_RE.search(unit_stripped)
        if not m:
            return _fail(
                "Uncited claim unit (missing end-of-sentence citation)",
                {"unit": unit},
            )

        cid = m.group(1)
        if cid not in allowed:
            return _fail(
                "Invalid citation ID",
                {"unit": unit, "citation_id": cid, "allowed": sorted(allowed)},
            )

    return _pass()

