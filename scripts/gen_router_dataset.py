#!/usr/bin/env python3
import json
import random
from pathlib import Path

# Adjust if needed
VAULT_ROOT = None  # leave None to use src.config.VAULT_PATH
LINK_INDEX_PATH = Path("data/index/links.json")
OUT_PATH = Path("data/router_train.jsonl")

random.seed(7)

BROWSE_ALIASES = {
    "projects": "03_projects",
    "project": "03_projects",
    "areas": "02_areas",
    "area": "02_areas",
    "resources": "04_resources",
    "resource": "04_resources",
    "archive": "05_archive",
    "inbox": "00_inbox",
    "templates": "99_templates",
}

BROWSE_TEMPLATES = [
    "which documents reside in {folder} folder ?",
    "list documents in {folder} folder",
    "show files under {folder} folder",
    "what documents are inside {folder} folder?",
    "which notes are in {folder} folder?",
]

BACKLINK_TEMPLATES = [
    "which notes link to {note} note?",
    "who links to {note}?",
    "backlinks of {note}?",
    "which files reference {note} note?",
]

OUTGOING_TEMPLATES = [
    "what does {note} link to?",
    "what {note}.md links to ?",
    "outgoing links of {note}?",
    "what links are in {note} note?",
]

DOC_EXTRACT_TEMPLATES = [
    "extract resources from {note} document",
    "what resources are mentioned in {note} note?",
    "list the resources in {note}",
]

RAG_QA_TEMPLATES = [
    "why smart pointers?",
    "what is FAISS?",
    "how to avoid distraction?",
    "what is retrieval augmented generation?",
]

def load_vault_path():
    if VAULT_ROOT is not None:
        return Path(VAULT_ROOT)
    from src.config import VAULT_PATH
    return Path(VAULT_PATH)

def note_name_from_rel_path(rel_path: str) -> str:
    # "path/to/Foo.md" -> "Foo"
    p = Path(rel_path)
    return p.stem

def plan_backlinks(note: str) -> dict:
    return {
        "intent": "DOC_EXTRACT",
        "target_note": note,
        "actions": [{"tool": "backlinks", "args": {"target_note": note, "limit": 100}}],
    }

def plan_outgoing(note: str) -> dict:
    return {
        "intent": "DOC_EXTRACT",
        "target_note": note,
        "actions": [
            {"tool": "resolve_note", "args": {"name": note}},
            {"tool": "outgoing_links", "args": {}},
        ],
    }

def plan_browse(prefix: str, include_files: bool = True, include_dirs: bool = False, depth: int = 2) -> dict:
    return {
        "intent": "BROWSE",
        "target_note": None,
        "actions": [{
            "tool": "browse_vault",
            "args": {
                "prefix": prefix,
                "depth": depth,
                "include_dirs": include_dirs,
                "include_files": include_files,
                "only_ext": [".md"],
            }
        }]
    }

def plan_doc_extract(note: str) -> dict:
    return {
        "intent": "DOC_EXTRACT",
        "target_note": note,
        "actions": [
            {"tool": "resolve_note", "args": {"name": note}},
            {"tool": "read_note", "args": {}},
            {"tool": "extract_resources", "args": {}},
        ],
    }

def plan_rag_qa(q: str) -> dict:
    return {
        "intent": "RAG_QA",
        "target_note": None,
        "actions": [{"tool": "search", "args": {"query": q, "top_k": 5}}],
    }

def list_folders(vault_root: Path, max_depth: int = 6):
    # Collect folder rel_paths (vault-relative)
    out = []
    base_parts = len(vault_root.parts)

    for p in vault_root.rglob("*"):
        if not p.is_dir():
            continue
        rel = p.relative_to(vault_root)
        depth = len(p.parts) - base_parts
        if depth <= 0 or depth > max_depth:
            continue
        rel_s = str(rel).replace("\\", "/")
        out.append(rel_s)

    out = sorted(set(out))
    return out

def main():
    vault_root = load_vault_path()

    if not LINK_INDEX_PATH.exists():
        raise SystemExit(f"Missing {LINK_INDEX_PATH}. Run: python -m src.embed_index")

    links = json.loads(LINK_INDEX_PATH.read_text(encoding="utf-8"))
    backlinks = links.get("backlinks", {}) or {}
    outgoing = links.get("outgoing", {}) or links.get("outgoing_links", {}) or {}

    # 1) Backlink targets: keys that actually have incoming refs
    backlink_targets = [k for k, v in backlinks.items() if isinstance(v, list) and len(v) > 0]

    # 2) Outgoing sources: rel_paths that have outgoing links
    outgoing_sources = []
    for rel_path, v in outgoing.items():
        if not isinstance(v, dict):
            continue
        wl = v.get("wikilinks", []) or []
        ml = v.get("mdlinks", []) or []
        if wl or ml:
            outgoing_sources.append(rel_path)

    # 3) Folders for browsing
    folders = list_folders(vault_root, max_depth=6)

    # Also add PARA aliases (“Projects folder” -> 03_projects)
    alias_folders = list(BROWSE_ALIASES.keys())

    examples = []

    # --- Generate BROWSE examples ---
    # a) alias-based
    for alias in alias_folders:
        q = random.choice(BROWSE_TEMPLATES).format(folder=alias.capitalize())
        prefix = BROWSE_ALIASES[alias]
        examples.append((q, plan_browse(prefix, include_files=True, include_dirs=False, depth=2)))

    # b) real folder relpaths (sample)
    for folder in random.sample(folders, min(120, len(folders))):
        # use leaf folder name in question (your browse_vault now resolves bare names)
        leaf = folder.split("/")[-1]
        q = random.choice(BROWSE_TEMPLATES).format(folder=leaf)
        # still use leaf in query to train “folder name resolution” routing, but plan uses browse_vault with that leaf
        examples.append((q, plan_browse(leaf, include_files=True, include_dirs=False, depth=2)))

    # --- Generate BACKLINK examples ---
    for note in random.sample(backlink_targets, min(120, len(backlink_targets))):
        q = random.choice(BACKLINK_TEMPLATES).format(note=note)
        examples.append((q, plan_backlinks(note)))

    # --- Generate OUTGOING examples ---
    for rel_path in random.sample(outgoing_sources, min(120, len(outgoing_sources))):
        note = note_name_from_rel_path(rel_path)
        q = random.choice(OUTGOING_TEMPLATES).format(note=note)
        examples.append((q, plan_outgoing(note)))

    # --- Generate DOC_EXTRACT resource-extraction examples ---
    # Use backlink targets as note names (cheap pool)
    for note in random.sample(backlink_targets, min(80, len(backlink_targets))):
        q = random.choice(DOC_EXTRACT_TEMPLATES).format(note=note)
        examples.append((q, plan_doc_extract(note)))

    # --- Add small RAG_QA set ---
    for q in RAG_QA_TEMPLATES:
        examples.append((q, plan_rag_qa(q)))

    # Deduplicate by question text (keep first)
    seen = set()
    final = []
    for q, p in examples:
        qn = " ".join(q.strip().split())
        if qn in seen:
            continue
        seen.add(qn)
        final.append((qn, p))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for q, plan in final:
            rec = {"user_q": q, "target_json": json.dumps(plan, ensure_ascii=False)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final)} examples -> {OUT_PATH}")

if __name__ == "__main__":
    main()
