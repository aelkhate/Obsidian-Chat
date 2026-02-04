# src/orchestrator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .router_agent import route, Plan
from .tools import (
    search,
    resolve_note,
    read_note,
    extract_resources,
    browse_vault,
    backlinks,
    outgoing_links,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .local_llm import LocalLLM
from .verifier import verify_answer
from .trace_logger import append_jsonl, build_trace, make_run_id


REFUSAL_TEXT = "I couldn't find that in your notes."
TRACE_PATH_DEFAULT = "logs/traces.jsonl"


def _format_sources(results: List[Dict[str, Any]], max_chars_per_source: int = 900) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        text = (r.get("text") or "").strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."
        blocks.append(
            f"[{i}] rel_path: {r['rel_path']}\n"
            f"    heading: {r.get('heading','ROOT')}\n"
            f"    chunk_id: {r.get('chunk_id','')}\n"
            f"    score: {float(r.get('score', 0.0)):.4f}\n"
            f"    text:\n{text}\n"
        )
    return "\n".join(blocks)


def _observer_verify_and_maybe_repair(
    *,
    plan: Plan,
    llm: LocalLLM,
    user_q: str,
    answer: str,
    sources_for_verifier: List[Dict[str, Any]],
    sources_block_for_llm: str,
) -> Dict[str, Any]:
    """
    Observer mode:
    - Run verifier
    - If FAIL, do ONE repair attempt to improve formatting/citations
    - Return best-effort answer either way (DO NOT BLOCK)
    """
    verdict1 = verify_answer(answer, sources_for_verifier, plan.intent)
    if verdict1["status"] == "PASS":
        return {
            "answer": answer,
            "verifier": verdict1,
            "repaired": False,
            "repaired_answer": None,
        }

    # One repair attempt (helps training data quality + reduces FAIL rate)
    repair_prompt = (
        sources_block_for_llm
        + "\n\nREPAIR INSTRUCTIONS:\n"
        + "- Your previous answer FAILED verification.\n"
        + "- Rewrite the answer so that EVERY sentence (and EVERY bullet item) ends with exactly one citation like (source: [n]).\n"
        + "- The citation must appear BEFORE the final punctuation. For ':' lead-ins, write: '... include (source: [1]):'\n"
        + "- For normal sentences use: ... (source: [1]).\n"
        + "- Use ONLY the provided SOURCES.\n"
        + f'- If the SOURCES are insufficient, output exactly: "{REFUSAL_TEXT}"\n'
    )

    repair_user_prompt = build_user_prompt(user_q, repair_prompt)
    repaired_answer = llm.generate(SYSTEM_PROMPT, repair_user_prompt, max_new_tokens=500)

    verdict2 = verify_answer(repaired_answer, sources_for_verifier, plan.intent)

    # Observer mode: even if still FAIL, we return the repaired answer (often closer),
    # but we keep verifier status for logs/eval.
    # If the repair made it worse (rare), keep original.
    use_answer = repaired_answer if len(repaired_answer.strip()) >= 1 else answer
    use_verdict = verdict2 if verdict2 else verdict1

    return {
        "answer": use_answer,
        "verifier": use_verdict,
        "repaired": True,
        "repaired_answer": repaired_answer,
        "verifier_before_repair": verdict1,
    }


def run_query(
    user_q: str,
    llm: LocalLLM,
    *,
    trace_path: str = TRACE_PATH_DEFAULT,
    enable_trace: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = run_id or make_run_id()
    plan: Plan = route(user_q, llm=llm)

    evidence: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []
    extracted: Dict[str, List[str]] | None = None

    resolved = None
    markdown: str | None = None

    # -------------------------
    # Execute plan actions only
    # -------------------------
    for act in plan.actions:
        if act.tool == "search":
            res = search(act.args["query"], top_k=act.args.get("top_k", 5))
            sources = res

        elif act.tool == "resolve_note":
            name = act.args.get("name") or plan.target_note or user_q
            candidates = resolve_note(name, limit=5)
            evidence["resolve_note"] = candidates
            resolved = candidates[0] if candidates else None

        elif act.tool == "read_note":
            if not resolved:
                markdown = None
            else:
                markdown = read_note(resolved["rel_path"])
                evidence["read_note_rel_path"] = resolved["rel_path"]

        elif act.tool == "backlinks":
            target = act.args.get("target_note") or plan.target_note or user_q
            limit = int(act.args.get("limit", 100))
            bl = backlinks(target, limit=limit)
            evidence["backlinks"] = {"target": target, "results": bl}

        elif act.tool == "outgoing_links":
            # Prefer explicit rel_path arg, else use read_note evidence, else fall back to resolved note.
            rel_path = act.args.get("rel_path") or evidence.get("read_note_rel_path")
            if not rel_path and resolved and isinstance(resolved, dict):
                rel_path = resolved.get("rel_path")

            if rel_path:
                ol = outgoing_links(rel_path)
                evidence["outgoing_links"] = {"rel_path": rel_path, "links": ol}


        elif act.tool == "browse_vault":
            prefix = act.args.get("prefix", "03_projects")
            depth = int(act.args.get("depth", 2))
            include_dirs = bool(act.args.get("include_dirs", True))
            include_files = bool(act.args.get("include_files", False))
            only_ext = act.args.get("only_ext", [".md"])

            browsed = browse_vault(
                prefix=prefix,
                depth=depth,
                include_dirs=include_dirs,
                include_files=include_files,
                only_ext=only_ext,
            )

            evidence["browse_vault"] = {
                "prefix": prefix,
                "depth": depth,
                "include_dirs": include_dirs,
                "include_files": include_files,
                "only_ext": only_ext,
                "results": browsed,
            }

        elif act.tool == "extract_resources":
            if markdown:
                extracted = extract_resources(markdown)
                evidence["extract_resources"] = extracted

        else:
            evidence.setdefault("unknown_tools", []).append({"tool": act.tool, "args": act.args})

    # -------------------------
    # Answer construction (grounded)
    # -------------------------

    # ---- BROWSE
    if plan.intent == "BROWSE" and "browse_vault" in evidence:
        b = evidence["browse_vault"]
        results = b.get("results", [])
        if not results:
            # Deterministic, explicit explanation instead of blind refusal
            prefix = b.get("prefix", "UNKNOWN")
            depth = b.get("depth", None)

            msg_lines = [
                f"No items were found under `{prefix}`."
            ]
            if depth is not None:
                msg_lines.append(f"Browse depth was limited to {depth}.")
            msg_lines.append("This means there are no matching notes or folders at that location.")

            answer = " ".join(msg_lines)

            out = {
                "intent": plan.intent,
                "plan": plan.model_dump(),
                "answer": answer,
                "sources": [],
                "verifier": None,   # browse output is deterministic text, no factual claims
            }

            if enable_trace:
                append_jsonl(
                    trace_path,
                    build_trace(
                        run_id=run_id,
                        user_q=user_q,
                        intent=plan.intent,
                        plan=out.get("plan"),
                        answer=out["answer"],
                        sources=out["sources"],
                        verifier=None,
                        notes={
                            "reason": "browse_empty",
                            "prefix": prefix,
                            "depth": depth,
                        },
                    ),
                )

            return out


        lines = [f"PREFIX: {b.get('prefix')}", f"DEPTH: {b.get('depth')}"]
        for r in results:
            lines.append(f"- {r['kind']}: {r['rel_path']}")
        inv_blob = "\n".join(lines)

        sources_block = (
            "[1] rel_path: "
            + b.get("prefix", "UNKNOWN")
            + "\n    heading: BROWSE_INVENTORY\n"
            + "    chunk_id: BROWSE::INVENTORY\n"
            + "    score: 1.0000\n"
            + "    text:\n"
            + inv_blob
        )

        user_prompt = build_user_prompt(user_q, sources_block)
        raw_answer = llm.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens=500)

        resolved_prefix = evidence["browse_vault"].get("resolved_prefix") or b.get("prefix", "UNKNOWN")
        sources_out = [{"rel_path": resolved_prefix, "heading": "BROWSE_INVENTORY", "score": 1.0}]

        obs = _observer_verify_and_maybe_repair(
            plan=plan,
            llm=llm,
            user_q=user_q,
            answer=raw_answer,
            sources_for_verifier=sources_out,
            sources_block_for_llm=sources_block,
        )

        out = {
            "intent": plan.intent,
            "plan": plan.model_dump(),
            "answer": obs["answer"],
            "sources": sources_out,
            "verifier": obs["verifier"],
            "repaired": obs.get("repaired", False),
        }
        if "verifier_before_repair" in obs:
            out["verifier_before_repair"] = obs["verifier_before_repair"]

        if enable_trace:
            append_jsonl(trace_path, build_trace(
                run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                answer=out["answer"], sources=out["sources"], verifier=out.get("verifier"),
                repaired=out.get("repaired", False),
                repaired_answer=obs.get("repaired_answer"),
                notes={"mode": "observer"},
            ))
        return out

    # ---- DOC_EXTRACT (resources extraction)
    if plan.intent == "DOC_EXTRACT" and extracted:
        doc_text: List[str] = []
        for heading, items in extracted.items():
            doc_text.append(f"## {heading}")
            for it in items:
                doc_text.append(f"- {it}")
        doc_blob = "\n".join(doc_text)

        relp = evidence.get("read_note_rel_path", "UNKNOWN")
        sources_block = (
            "[1] rel_path: "
            + relp
            + "\n    heading: FULL_DOCUMENT\n"
            + "    chunk_id: DOC_EXTRACT::FULL\n"
            + "    score: 1.0000\n"
            + "    text:\n"
            + doc_blob
        )

        user_prompt = build_user_prompt(user_q, sources_block)
        raw_answer = llm.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens=500)

        sources_out = [{"rel_path": relp, "heading": "FULL_DOCUMENT", "score": 1.0}]
        obs = _observer_verify_and_maybe_repair(
            plan=plan,
            llm=llm,
            user_q=user_q,
            answer=raw_answer,
            sources_for_verifier=sources_out,
            sources_block_for_llm=sources_block,
        )

        out = {
            "intent": plan.intent,
            "plan": plan.model_dump(),
            "answer": obs["answer"],
            "sources": sources_out,
            "verifier": obs["verifier"],
            "repaired": obs.get("repaired", False),
        }
        if "verifier_before_repair" in obs:
            out["verifier_before_repair"] = obs["verifier_before_repair"]

        if enable_trace:
            append_jsonl(trace_path, build_trace(
                run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                answer=out["answer"], sources=out["sources"], verifier=out.get("verifier"),
                repaired=out.get("repaired", False),
                repaired_answer=obs.get("repaired_answer"),
                notes={"mode": "observer"},
            ))
        return out

    # ---- LINK QUERIES (deterministic answers; verifier optional)
    if plan.intent in ("DOC_EXTRACT", "SYNTHESIS") and ("backlinks" in evidence or "outgoing_links" in evidence):
        # backlinks
        if "backlinks" in evidence:
            bl = evidence["backlinks"]
            target = bl.get("target")
            results = bl.get("results", []) or []

            if not results:
                out = {"intent": plan.intent, "plan": plan.model_dump(), "answer": REFUSAL_TEXT, "sources": []}
                if enable_trace:
                    append_jsonl(trace_path, build_trace(
                        run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                        answer=out["answer"], sources=out["sources"], verifier=None,
                        notes={"reason": "backlinks_empty"},
                    ))
                return out

            # deterministic answer (no LLM)
            sources_out = [{"rel_path": "data/index/links.json", "heading": "BACKLINKS", "score": 1.0}]
            lines = [f'Notes that link to "{target}" (source: [1]).']
            for p in results:
                lines.append(f"- {p} (source: [1]).")
            answer = "\n".join(lines)

            verifier = verify_answer(answer, sources_out, plan.intent)

            out = {
                "intent": plan.intent,
                "plan": plan.model_dump(),
                "answer": answer,
                "sources": sources_out,
                "verifier": verifier,
                "repaired": False,
            }
            if enable_trace:
                append_jsonl(trace_path, build_trace(
                    run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                    answer=out["answer"], sources=out["sources"], verifier=out.get("verifier"),
                    notes={"mode": "observer", "deterministic": True},
                ))
            return out

        # outgoing links
        if "outgoing_links" in evidence:
            ol = evidence["outgoing_links"]
            rel_path = ol.get("rel_path")
            links = ol.get("links", {}) or {}
            wikilinks = links.get("wikilinks", []) or []
            mdlinks = links.get("mdlinks", []) or []

            if not wikilinks and not mdlinks:
                out = {"intent": plan.intent, "plan": plan.model_dump(), "answer": REFUSAL_TEXT, "sources": []}
                if enable_trace:
                    append_jsonl(trace_path, build_trace(
                        run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                        answer=out["answer"], sources=out["sources"], verifier=None,
                        notes={"reason": "outgoing_empty"},
                    ))
                return out

            sources_out = [{"rel_path": rel_path or "UNKNOWN", "heading": "OUTGOING_LINKS", "score": 1.0}]
            lines: List[str] = [f"Outgoing links in {rel_path} (source: [1])."]
            if wikilinks:
                lines.append("## Wikilinks")
                for x in wikilinks:
                    lines.append(f"- {x} (source: [1]).")
            if mdlinks:
                lines.append("## Markdown links")
                for x in mdlinks:
                    lines.append(f"- {x} (source: [1]).")
            answer = "\n".join(lines)

            verifier = verify_answer(answer, sources_out, plan.intent)

            out = {
                "intent": plan.intent,
                "plan": plan.model_dump(),
                "answer": answer,
                "sources": sources_out,
                "verifier": verifier,
                "repaired": False,
            }
            if enable_trace:
                append_jsonl(trace_path, build_trace(
                    run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                    answer=out["answer"], sources=out["sources"], verifier=out.get("verifier"),
                    notes={"mode": "observer", "deterministic": True},
                ))
            return out

    # ---- Default: RAG_QA (search-based)
    if not sources:
        out = {"intent": plan.intent, "plan": plan.model_dump(), "answer": REFUSAL_TEXT, "sources": []}
        if enable_trace:
            append_jsonl(trace_path, build_trace(
                run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
                answer=out["answer"], sources=out["sources"], verifier=None,
                notes={"reason": "search_empty"},
            ))
        return out

    sources_block = _format_sources(sources)
    user_prompt = build_user_prompt(user_q, sources_block)
    raw_answer = llm.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens=500)

    obs = _observer_verify_and_maybe_repair(
        plan=plan,
        llm=llm,
        user_q=user_q,
        answer=raw_answer,
        sources_for_verifier=sources,
        sources_block_for_llm=sources_block,
    )

    out = {
        "intent": plan.intent,
        "plan": plan.model_dump(),
        "answer": obs["answer"],
        "sources": sources,
        "verifier": obs["verifier"],
        "repaired": obs.get("repaired", False),
    }
    if "verifier_before_repair" in obs:
        out["verifier_before_repair"] = obs["verifier_before_repair"]

    if enable_trace:
        append_jsonl(trace_path, build_trace(
            run_id=run_id, user_q=user_q, intent=plan.intent, plan=out.get("plan"),
            answer=out["answer"], sources=out["sources"], verifier=out.get("verifier"),
            repaired=out.get("repaired", False),
            repaired_answer=obs.get("repaired_answer"),
            notes={"mode": "observer"},
        ))

    return out
