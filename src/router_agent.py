# src/router_agent.py
import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .local_llm import LocalLLM


Intent = Literal["RAG_QA", "DOC_EXTRACT", "BROWSE", "SYNTHESIS"]

_FOLDER_Q_RE = re.compile(
    r"""
    (?:
        (?:what|which)\s+(?:documents|files|notes)\s+(?:reside|are|exist|live)\s+in\s+
        |
        (?:list|show)\s+(?:documents|files|notes)\s+(?:in|under)\s+
        |
        (?:browse)\s+
    )
    (?P<prefix>.+?)
    (?:\s+(?:folder|directory|dir))?\s*[\?\.!]*\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _clean_prefix(s: str) -> str:
    s = (s or "").strip().strip(' "\'')
    s = re.sub(r"\s+", " ", s)
    # remove trailing "folder"/"directory" if still present
    s = re.sub(r"\b(folder|directory|dir)\b\s*$", "", s, flags=re.IGNORECASE).strip()
    return s


class Action(BaseModel):
    tool: Literal[
        "search",
        "resolve_note",
        "read_note",
        "extract_resources",
        "browse_vault",
        "backlinks",
        "outgoing_links",
    ]
    args: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    intent: Intent
    target_note: Optional[str] = None
    actions: List[Action]


ROUTER_SYSTEM = """You are a Router Agent for an Obsidian assistant.
Your job is to output a JSON plan using ONLY the allowed intents and tools.

Allowed intents:
- RAG_QA: explain/answer based on retrieved chunks
- DOC_EXTRACT: user asks to list/extract/summarize content from a specific document/note OR extract structured info from a note
- BROWSE: user asks about what notes exist / what projects / navigation / inventory
- SYNTHESIS: user asks to combine across multiple notes into a plan or comparison

Allowed tools:
- search(query, top_k)
- resolve_note(name)
- read_note(rel_path)
- extract_resources(markdown)
- browse_vault(prefix, depth, include_dirs, include_files, only_ext)
- backlinks(target_note, limit)
- outgoing_links(rel_path)

Output MUST be valid JSON and MUST match this schema:
{
  "intent": "...",
  "target_note": "optional string",
  "actions": [{"tool": "...", "args": {...}}, ...]
}

Rules (BROWSE):
- If the user asks what exists in the vault (projects/areas/resources/archive), navigation, inventory, "what notes/files", choose BROWSE.
- For BROWSE: plan a browse_vault action with an appropriate prefix.
  * Active projects: prefix="03_projects".
  * Areas: prefix="02_areas".
  * Resources: prefix="04_resources".
  * Archive: prefix="05_archive".
- Use depth=2 by default unless the user asks for deeper.
- For listing projects/areas/resources, use include_dirs=True, include_files=False unless the user asks for files.
- If the user asks about what documents/files/notes are inside a folder/directory/prefix, choose BROWSE and use browse_vault(prefix=...) with include_files=True.

Rules (LINKS):
- BACKLINKS (incoming links):
  If the user asks "which notes link to X" / "who links to X" / "backlinks of X", choose DOC_EXTRACT.
  Plan: backlinks(target_note=<X>, limit=100).

- OUTGOING LINKS (forward links):
  If the user asks "what does X link to" / "what links are in X" / "outgoing links of X", choose DOC_EXTRACT.
  Plan: resolve_note(name=<X>) then outgoing_links(rel_path=<resolved path>).

Rules (DOC_EXTRACT):
- If the user references a specific document (e.g., "in C++ Resources document") or asks to list/extract items from a note, choose DOC_EXTRACT.
- For DOC_EXTRACT (resources extraction): resolve_note(name) → read_note(rel_path) → extract_resources(markdown).

Rules (RAG_QA):
- Only if no other rule applies, for general questions (why/how/what is...), choose RAG_QA and use search(query, top_k=5).

General:
- Do NOT include any text outside the JSON.

Domain policy (PARA):
- 03_projects = active projects
- 02_areas = ongoing responsibilities
- 04_resources = reference material
- 05_archive = inactive/completed items
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
        "q": "what active projects do I currently have?",
        "plan": {
            "intent": "BROWSE",
            "actions": [
                {
                    "tool": "browse_vault",
                    "args": {
                        "prefix": "03_projects",
                        "depth": 2,
                        "include_dirs": True,
                        "include_files": False,
                        "only_ext": [".md"],
                    },
                }
            ],
        },
    },
    {
        "q": "which notes link to my C++ Resources note?",
        "plan": {
            "intent": "DOC_EXTRACT",
            "target_note": "C++ Resources",
            "actions": [{"tool": "backlinks", "args": {"target_note": "C++ Resources", "limit": 100}}],
        },
    },
    {
        "q": "what does my C++ Resources note link to?",
        "plan": {
            "intent": "DOC_EXTRACT",
            "target_note": "C++ Resources",
            "actions": [
                {"tool": "resolve_note", "args": {"name": "C++ Resources"}},
                {"tool": "outgoing_links", "args": {}},
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


def _strip_md_suffix(s: str) -> str:
    return re.sub(r"\.md\s*$", "", (s or "").strip(), flags=re.IGNORECASE).strip()


def _extract_target_after_to(user_q: str) -> str:
    m = re.search(r"\bto\b(.+)", user_q, flags=re.IGNORECASE)
    if not m:
        return user_q.strip().strip('?"\' ')
    return m.group(1).strip().strip('?"\' ')


def deterministic_fallback(user_q: str) -> Plan:
    import re

    raw = user_q.strip()
    q = raw.lower()

    def strip_md(s: str) -> str:
        return re.sub(r"\.md\s*$", "", (s or "").strip(), flags=re.IGNORECASE).strip()

    def extract_after_to(s: str) -> str:
        m = re.search(r"\bto\b(.+)", s, flags=re.IGNORECASE)
        if not m:
            return s.strip().strip('?"\' ')
        return m.group(1).strip().strip('?"\' ')

    # -------------------------
    # 1) BROWSE: explicit folder/directory inventory questions
    # -------------------------
    # Examples:
    # - "what documents reside in Sweet500Handover folder ?"
    # - "list files in 03_projects/Sweet500Handover"
    # - "show notes under sweet500Handover directory"
    folder_re = re.compile(
        r"""
        (?:
            (?:what|which)\s+(?:documents|files|notes)\s+(?:reside|are|exist|live)\s+in\s+
            |
            (?:list|show)\s+(?:documents|files|notes)\s+(?:in|under)\s+
            |
            (?:browse)\s+
        )
        (?P<prefix>.+?)
        (?:\s+(?:folder|directory|dir))?\s*[\?\.!]*\s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    m = folder_re.search(raw)
    if m:
        prefix = (m.group("prefix") or "").strip().strip(' "\'')
        prefix = re.sub(r"\s+", " ", prefix).strip()
        prefix = re.sub(r"\b(folder|directory|dir)\b\s*$", "", prefix, flags=re.IGNORECASE).strip()

        # If user asked for documents/files/notes -> include_files=True
        include_files = True
        include_dirs = True  # keep dirs too (structure is useful)
        return Plan(
            intent="BROWSE",
            actions=[
                Action(
                    tool="browse_vault",
                    args={
                        "prefix": prefix,
                        "depth": 2,
                        "include_dirs": include_dirs,
                        "include_files": include_files,
                        "only_ext": [".md"],
                    },
                )
            ],
        )

    # -------------------------
    # 2) OUTGOING LINKS: "what does X link to?"
    # -------------------------
    # MUST come before backlinks, because "links to" is ambiguous.
    if (
        re.search(r"\b(outgoing links|forward links)\b", q)
        or re.search(r"\bwhat\s+does\s+.+\s+link\s+to\b", q)
        or re.search(r"\bwhat\s+.+\s+links?\s+to\b", q)
        or re.search(r"\bwhat\s+links?\s+are\s+in\b", q)
    ):
        if " to " in q:
            target = extract_after_to(raw)
        else:
            target = re.sub(r"^\s*what\s+does\s+", "", raw, flags=re.IGNORECASE).strip()
            target = re.sub(r"\s+link\s+to\s*\??\s*$", "", target, flags=re.IGNORECASE).strip()
            target = re.sub(r"^\s*outgoing links (in|of)\s+", "", target, flags=re.IGNORECASE).strip()

        target = strip_md(target)
        return Plan(
            intent="DOC_EXTRACT",
            target_note=target,
            actions=[
                Action(tool="resolve_note", args={"name": target}),
                Action(tool="outgoing_links", args={}),
            ],
        )

    # -------------------------
    # 3) BACKLINKS: "which notes link to X?"
    # -------------------------
    if any(
        k in q
        for k in [
            "backlink",
            "backlinks",
            "which notes link",
            "which notes are linked",
            "who links to",
        ]
    ):
        target = strip_md(extract_after_to(raw))
        return Plan(
            intent="DOC_EXTRACT",
            target_note=target,
            actions=[Action(tool="backlinks", args={"target_note": target, "limit": 100})],
        )

    # -------------------------
    # 4) General BROWSE triggers (PARA navigation / inventory)
    # -------------------------
    if any(
        k in q
        for k in [
            "what projects",
            "active projects",
            "list projects",
            "show projects",
            "browse",
            "what notes",
            "what files",
            "inventory",
            "what areas",
            "areas",
            "what resources",
            "resources",
            "archive",
        ]
    ):
        prefix = "03_projects"
        if "areas" in q:
            prefix = "02_areas"
        elif "resources" in q:
            prefix = "04_resources"
        elif "archive" in q:
            prefix = "05_archive"

        # default inventory behavior: show dirs (structure) not files
        return Plan(
            intent="BROWSE",
            actions=[
                Action(
                    tool="browse_vault",
                    args={
                        "prefix": prefix,
                        "depth": 2,
                        "include_dirs": True,
                        "include_files": False,
                        "only_ext": [".md"],
                    },
                )
            ],
        )

    # -------------------------
    # 5) DOC_EXTRACT triggers (resources / list everything in a note)
    # -------------------------
    if (
        ("in " in q and "document" in q)
        or ("mentioned in" in q)
        or ("summarize" in q and "note" in q)
        or ("from the" in q and "note" in q)
    ):
        return Plan(
            intent="DOC_EXTRACT",
            target_note=raw,
            actions=[
                Action(tool="resolve_note", args={"name": raw}),
                Action(tool="read_note", args={}),
                Action(tool="extract_resources", args={}),
            ],
        )

    # -------------------------
    # 6) default: QA
    # -------------------------
    return Plan(
        intent="RAG_QA",
        actions=[Action(tool="search", args={"query": raw, "top_k": 5})],
    )


def route(user_q: str, llm: Optional[LocalLLM] = None) -> Plan:
    llm = llm or LocalLLM()
    raw = llm.generate(ROUTER_SYSTEM, _router_prompt(user_q), max_new_tokens=350)

    try:
        data = json.loads(raw)
        plan = Plan(**data)
        return plan
    except (json.JSONDecodeError, ValidationError):
        return deterministic_fallback(user_q)
