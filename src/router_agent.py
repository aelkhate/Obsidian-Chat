import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .local_llm import LocalLLM


Intent = Literal["RAG_QA", "DOC_EXTRACT", "BROWSE", "SYNTHESIS"]


class Action(BaseModel):
    tool: Literal["search", "resolve_note", "read_note", "extract_resources"]
    args: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    intent: Intent
    target_note: Optional[str] = None
    actions: List[Action]


ROUTER_SYSTEM = """You are a Router Agent for an Obsidian assistant.
Your job is to output a JSON plan using ONLY the allowed intents and tools.

Allowed intents:
- RAG_QA: explain/answer based on retrieved chunks
- DOC_EXTRACT: user asks to list/extract/summarize content from a specific document/note
- BROWSE: user asks about what notes exist / what projects / navigation
- SYNTHESIS: user asks to combine across multiple notes into a plan or comparison

Allowed tools:
- search(query, top_k)
- resolve_note(name)
- read_note(rel_path)
- extract_resources(markdown)

Output MUST be valid JSON and MUST match this schema:
{
  "intent": "...",
  "target_note": "optional string",
  "actions": [{"tool": "...", "args": {...}}, ...]
}

Rules:
- If the user references a specific document (e.g., "in C++ Resources document"), choose DOC_EXTRACT.
- For DOC_EXTRACT: first resolve_note(name), then read_note(rel_path), then extract_resources(markdown).
- For general questions (why/how/what is...), choose RAG_QA and use search(query, top_k=5).
- Do NOT include any text outside the JSON.

Domain policy (PARA):
- 03_projects = active projects
- 05_archive = inactive projects/resources
- "active projects" means: items under 03_projects (NOT based on file modified time).
"""


FEWSHOT = [
    {
        "q": "what are my C++ resources mentioned in C++ resources document?",
        "plan": {
            "intent": "DOC_EXTRACT",
            "target_note": "C++ Resources",
            "actions": [
                {"tool": "resolve_note", "args": {"name": "C++ Resources"}},
                {"tool": "read_note", "args": {}},
                {"tool": "extract_resources", "args": {}},
            ],
        },
    },
    {
        "q": "why smart pointers?",
        "plan": {
            "intent": "RAG_QA",
            "actions": [{"tool": "search", "args": {"query": "why smart pointers?", "top_k": 5}}],
        },
    },
]


def _router_prompt(user_q: str) -> str:
    examples = "\n\n".join(
        [
            f"Example Question: {ex['q']}\nExample Plan:\n{json.dumps(ex['plan'], ensure_ascii=False)}"
            for ex in FEWSHOT
        ]
    )
    return f"{examples}\n\nUser Question: {user_q}\nReturn ONLY the JSON plan."


def deterministic_fallback(user_q: str) -> Plan:
    q = user_q.lower()
    # DOC_EXTRACT triggers
    if ("in " in q and "document" in q) or ("mentioned in" in q) or ("summarize" in q and "note" in q) or ("from the" in q and "note" in q):
        # naive target extraction: let resolve_note handle fuzzy matching
        return Plan(
            intent="DOC_EXTRACT",
            target_note=user_q,  # pass full q; executor will use resolve_note anyway
            actions=[
                Action(tool="resolve_note", args={"name": user_q}),
                Action(tool="read_note", args={}),
                Action(tool="extract_resources", args={}),
            ],
        )

    # default: QA
    return Plan(
        intent="RAG_QA",
        actions=[Action(tool="search", args={"query": user_q, "top_k": 5})],
    )


def route(user_q: str, llm: Optional[LocalLLM] = None) -> Plan:
    llm = llm or LocalLLM()
    raw = llm.generate(ROUTER_SYSTEM, _router_prompt(user_q), max_new_tokens=350)

    # try parse JSON
    try:
        data = json.loads(raw)
        plan = Plan(**data)
        return plan
    except (json.JSONDecodeError, ValidationError):
        return deterministic_fallback(user_q)
