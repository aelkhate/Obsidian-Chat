from typing import Any, Dict, List, Tuple

from .router_agent import route, Plan
from .tools import search, resolve_note, read_note, extract_resources
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .local_llm import LocalLLM


def run_query(user_q: str, llm: LocalLLM) -> Dict[str, Any]:
    plan: Plan = route(user_q, llm=llm)

    evidence = {}
    sources = []
    extracted = None

    resolved = None
    markdown = None

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
                # nothing to read
                markdown = None
            else:
                markdown = read_note(resolved["rel_path"])
                evidence["read_note_rel_path"] = resolved["rel_path"]

        elif act.tool == "extract_resources":
            if markdown:
                extracted = extract_resources(markdown)
                evidence["extract_resources"] = extracted

    # Decide how to answer:
    if plan.intent == "DOC_EXTRACT" and extracted:
        # Turn extracted dict into SOURCES-like text (so Answer Agent is still grounded)
        # We'll cite a single "document source" using [1].
        doc_text = []
        for heading, items in extracted.items():
            doc_text.append(f"## {heading}")
            for it in items:
                doc_text.append(f"- {it}")
        doc_blob = "\n".join(doc_text)

        sources_block = (
            "[1] rel_path: "
            + evidence.get("read_note_rel_path", "UNKNOWN")
            + "\n    heading: FULL_DOCUMENT\n"
            + "    chunk_id: DOC_EXTRACT::FULL\n"
            + "    score: 1.0000\n"
            + "    text:\n"
            + doc_blob
        )

        user_prompt = build_user_prompt(user_q, sources_block)
        answer = llm.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens=500)

        return {
            "intent": plan.intent,
            "plan": plan.model_dump(),
            "answer": answer,
            "sources": [{"rel_path": evidence.get("read_note_rel_path", "UNKNOWN"), "heading": "FULL_DOCUMENT", "score": 1.0}],
        }

    # default: RAG_QA
    if not sources:
        return {
            "intent": plan.intent,
            "plan": plan.model_dump(),
            "answer": "I couldn't find that in your notes.",
            "sources": [],
        }

    # Build normal prompt from retrieved chunks
    # Keep your existing formatting function in chat.py or replicate here
    def format_sources(results, max_chars_per_source=900) -> str:
        blocks = []
        for i, r in enumerate(results, 1):
            text = r["text"].strip()
            if len(text) > max_chars_per_source:
                text = text[:max_chars_per_source].rstrip() + "..."
            blocks.append(
                f"[{i}] rel_path: {r['rel_path']}\n"
                f"    heading: {r['heading']}\n"
                f"    chunk_id: {r['chunk_id']}\n"
                f"    score: {r['score']:.4f}\n"
                f"    text:\n{text}\n"
            )
        return "\n".join(blocks)

    sources_block = format_sources(sources)
    user_prompt = build_user_prompt(user_q, sources_block)
    answer = llm.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens=500)

    return {
        "intent": plan.intent,
        "plan": plan.model_dump(),
        "answer": answer,
        "sources": sources,
    }
