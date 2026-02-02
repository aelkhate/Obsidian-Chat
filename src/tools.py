import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import process, fuzz

from .config import VAULT_PATH
from .retrieve import retrieve
from .catalog import build_note_catalog


# -------------------------
# NOTE RESOLUTION
# -------------------------

def browse_vault(
    prefix: str,
    depth: int = 1,
    include_dirs: bool = True,
    include_files: bool = False,
    only_ext: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generic vault browser.

    Args:
        prefix: relative path inside the vault (e.g. "03_projects")
        depth: how deep to traverse from prefix
        include_dirs: include directories
        include_files: include files
        only_ext: optional list of file extensions (e.g. [".md"])

    Returns:
        List of entries with rel_path and kind.
    """

    base_abs = os.path.join(VAULT_PATH, prefix)
    base_abs = os.path.normpath(base_abs)

    if not os.path.isdir(base_abs):
        return []

    results: List[Dict[str, Any]] = []

    base_depth = base_abs.count(os.sep)

    for root, dirs, files in os.walk(base_abs):
        cur_depth = root.count(os.sep) - base_depth
        if cur_depth >= depth:
            dirs[:] = []  # stop deeper traversal

        rel_root = os.path.relpath(root, VAULT_PATH)
        if rel_root == ".":
            rel_root = ""

        if include_dirs:
            for d in dirs:
                rel_path = os.path.normpath(os.path.join(rel_root, d)).replace("\\", "/")
                results.append({
                    "rel_path": rel_path,
                    "kind": "dir",
                })

        if include_files:
            for f in files:
                if only_ext and not any(f.lower().endswith(ext) for ext in only_ext):
                    continue
                rel_path = os.path.normpath(os.path.join(rel_root, f)).replace("\\", "/")
                results.append({
                    "rel_path": rel_path,
                    "kind": "file",
                })

    return sorted(results, key=lambda x: x["rel_path"].lower())

def resolve_note(name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Resolve a user-provided note name to actual rel_path(s).
    Returns ranked candidates with a score.
    """
    catalog = build_note_catalog()

    # candidates: match against title and rel_path tail
    choices = []
    for n in catalog:
        title = n["title"]
        rel = n["rel_path"]
        tail = os.path.splitext(os.path.basename(rel))[0]
        key = f"{title} | {tail} | {rel}"
        choices.append((key, n))

    matches = process.extract(
        name,
        [c[0] for c in choices],
        scorer=fuzz.WRatio,
        limit=limit,
    )

    # map back to notes
    key_to_note = {c[0]: c[1] for c in choices}
    out = []
    for key, score, _ in matches:
        note = key_to_note[key]
        out.append({
            "rel_path": note["rel_path"],
            "title": note["title"],
            "score": float(score),
            "headings": note["headings"],
        })
    return out


# -------------------------
# READ NOTE
# -------------------------

def read_note(rel_path: str) -> str:
    """
    Reads the full markdown note from the vault.
    """
    abs_path = os.path.join(VAULT_PATH, rel_path)
    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -------------------------
# DOC EXTRACTION
# -------------------------

_heading_re = re.compile(r"^(#{1,6})\s+(.*)\s*$")
_link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_url_re = re.compile(r"(https?://\S+)")


def extract_resources(markdown: str) -> Dict[str, List[str]]:
    """
    Deterministic extractor:
    - groups items by markdown heading
    - extracts:
      - explicit markdown links [text](url)
      - raw URLs
      - list items text lines
    Returns: {heading: [items...]}
    """
    sections: Dict[str, List[str]] = {}
    current = "ROOT"
    sections[current] = []

    lines = markdown.splitlines()
    for line in lines:
        m = _heading_re.match(line.strip())
        if m:
            current = m.group(2).strip()
            sections.setdefault(current, [])
            continue

        # collect list items and links
        stripped = line.strip()
        if not stripped:
            continue

        # markdown links
        for text, url in _link_re.findall(stripped):
            sections[current].append(f"{text.strip()} -> {url.strip()}")

        # raw urls
        for url in _url_re.findall(stripped):
            # avoid duplicates with markdown links
            sections[current].append(url.strip())

        # list items
        if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("1. "):
            sections[current].append(stripped)

    # de-dup while preserving order
    clean: Dict[str, List[str]] = {}
    for h, items in sections.items():
        seen = set()
        out = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        if out:
            clean[h] = out

    return clean


# -------------------------
# SEARCH TOOL (RAG)
# -------------------------

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return retrieve(query, top_k=top_k)
