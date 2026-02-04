import os
import json
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
    depth: int = 2,
    include_dirs: bool = True,
    include_files: bool = False,
    only_ext: list[str] | None = None,
):
    """
    Deterministic filesystem inventory under the Obsidian vault.

    Accepts:
    - vault-relative prefixes: "04_resources/..."
    - absolute paths (Windows/Linux) inside the vault
    - absolute paths containing a PARA root segment (04_resources, 03_projects, ...)
    - bare folder names like "Sweet500Handover" (resolved via deterministic scan)
    - minor typos/formatting differences (deterministic fuzzy match)
    """
    import re
    from pathlib import Path
    from difflib import SequenceMatcher

    from .config import VAULT_PATH

    only_ext = only_ext or [".md"]
    only_ext_l = [e.lower() for e in only_ext]

    PARA_ROOTS = [
        "00_inbox",
        "02_areas",
        "03_projects",
        "04_resources",
        "05_archive",
        "99_templates",
    ]

    def norm_slashes(s: str) -> str:
        return (s or "").replace("\\", "/").strip()

    def strip_drive(p: str) -> str:
        # "C:/x/y" -> "/x/y"
        if len(p) >= 2 and p[1] == ":":
            return p[2:]
        return p

    def normalize_name(s: str) -> str:
        # strong normalization for fuzzy folder matching
        return "".join(c.lower() for c in (s or "") if c.isalnum())

    def to_vault_relative(p: str) -> str:
        p = norm_slashes(p)
        p = strip_drive(p)

        vault = norm_slashes(str(VAULT_PATH))
        vault = strip_drive(vault)
        if not vault.endswith("/"):
            vault += "/"

        # If absolute path inside vault, relativize
        if p.startswith(vault):
            return p[len(vault):].strip("/")

        # Heuristic: extract suffix starting at a PARA root folder
        pl = p.lower()
        for root in PARA_ROOTS:
            r = root.lower()
            idx = pl.find("/" + r + "/")
            if idx != -1:
                return p[idx + 1 :].strip("/")  # skip leading "/"
            if pl.startswith(r + "/") or pl == r:
                return p.strip("/")

        # Otherwise assume it's already relative
        return p.strip("/")

    vault_root = Path(VAULT_PATH)

    # Step A: convert absolute/relative to vault-relative candidate
    rel_prefix = to_vault_relative(prefix)

    # Step B: if rel_prefix is a bare folder name (no slash), resolve it inside vault
    if rel_prefix and ("/" not in rel_prefix):
        target_norm = normalize_name(rel_prefix)

        # 1) exact folder-name match (case-insensitive)
        exact = []
        for p in vault_root.rglob("*"):
            if p.is_dir() and p.name.lower() == rel_prefix.lower():
                exact.append(p)

        if len(exact) == 1:
            rel_prefix = str(exact[0].relative_to(vault_root)).replace("\\", "/")
        elif len(exact) > 1:
            # ambiguous exact matches: return candidate dirs
            out = [{"kind": "dir", "rel_path": str(p.relative_to(vault_root)).replace("\\", "/")} for p in exact[:50]]
            out.sort(key=lambda x: x["rel_path"])
            return out

        else:
            # 2) deterministic fuzzy match on folder names
            scored = []
            for p in vault_root.rglob("*"):
                if not p.is_dir():
                    continue
                score = SequenceMatcher(None, target_norm, normalize_name(p.name)).ratio()
                if score >= 0.82:  # threshold (tune later)
                    scored.append((score, p))

            scored.sort(key=lambda x: x[0], reverse=True)

            if len(scored) == 1:
                rel_prefix = str(scored[0][1].relative_to(vault_root)).replace("\\", "/")
            elif len(scored) > 1:
                # ambiguous fuzzy matches: return candidates as dirs (with score encoded in path for now)
                out = [
                    {"kind": "dir", "rel_path": str(p.relative_to(vault_root)).replace("\\", "/"), "score": round(score, 3)}
                    for score, p in scored[:50]
                ]
                return out
            else:
                return []

    target = vault_root / rel_prefix
    if not target.exists() or not target.is_dir():
        return []

    # Depth-limited inventory
    results = []
    base_parts = len(target.parts)

    for path in target.rglob("*"):
        rel_parts = len(path.parts) - base_parts
        if rel_parts <= 0 or rel_parts > depth:
            continue

        if path.is_dir():
            if include_dirs:
                results.append({"kind": "dir", "rel_path": str(path.relative_to(vault_root)).replace("\\", "/")})
            continue

        if include_files:
            ext = path.suffix.lower()
            if only_ext_l and ext not in only_ext_l:
                continue
            results.append({"kind": "file", "rel_path": str(path.relative_to(vault_root)).replace("\\", "/")})

    results.sort(key=lambda x: (x["kind"], x["rel_path"]))
    return results

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

# -------------------------
# LINK GRAPH
# -------------------------

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
MDLINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

def extract_links(markdown: str) -> Dict[str, List[str]]:
    """Deterministically extract outgoing links from markdown.

    Returns:
      {"wikilinks": [...], "mdlinks": [...]}

    Wikilinks:
      - [[Target]]
      - [[Target|alias]]
      - [[Target#Heading]]
    """
    text = markdown or ""

    wikilinks: List[str] = []
    for m in WIKILINK_RE.finditer(text):
        t = (m.group(1) or "").strip()
        t = re.sub(r"\s+", " ", t)
        if t:
            wikilinks.append(t)

    mdlinks: List[str] = []
    for m in MDLINK_RE.finditer(text):
        url = (m.group(1) or "").strip()
        if url:
            mdlinks.append(url)

    def uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return {"wikilinks": uniq(wikilinks), "mdlinks": uniq(mdlinks)}

def _load_link_index(index_path: str = "data/index/links.json") -> Dict[str, Any]:
    if not os.path.isfile(index_path):
        return {"outgoing": {}, "backlinks": {}}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

def backlinks(target_note: str, limit: int = 100) -> List[str]:
    """Return rel_paths of notes that link to the given target note.

    target_note can be a note title, filename stem, or a rel_path.
    """
    idx = _load_link_index()
    bl = idx.get("backlinks", {}) or {}

    t = (target_note or "").strip()

    # candidate keys: user text + resolved note title/stem
    cand: List[str] = []
    if t:
        cand.append(t)
        cand.append(os.path.splitext(os.path.basename(t))[0])

    try:
        res = resolve_note(t, limit=3) if t else []
    except Exception:
        res = []

    for r in res:
        title = (r.get("title") or "").strip()
        stem = os.path.splitext(os.path.basename(r.get("rel_path", "")))[0]
        if title:
            cand.append(title)
        if stem:
            cand.append(stem)

    # normalize + de-dupe keys
    keys: List[str] = []
    seen = set()
    for k in cand:
        kk = re.sub(r"\s+", " ", (k or "").strip())
        if not kk:
            continue
        low = kk.lower()
        if low in seen:
            continue
        seen.add(low)
        keys.append(kk)

    out: List[str] = []
    seen_paths = set()
    for k in keys:
        for p in bl.get(k, []):
            if p in seen_paths:
                continue
            seen_paths.add(p)
            out.append(p)
            if len(out) >= limit:
                return out

    return out

def outgoing_links(rel_path: str) -> Dict[str, List[str]]:
    """Return outgoing links for a given note rel_path."""
    md = read_note(rel_path)
    return extract_links(md)

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    return retrieve(query, top_k=top_k)


def resolve_folder_fuzzy(name: str, threshold: float = 0.8):
    from pathlib import Path
    from difflib import SequenceMatcher
    from .config import VAULT_PATH

    def norm(s):
        return "".join(c.lower() for c in s if c.isalnum())

    vault_root = Path(VAULT_PATH)
    candidates = []

    target_norm = norm(name)

    for p in vault_root.rglob("*"):
        if not p.is_dir():
            continue

        folder = p.name
        score = SequenceMatcher(None, target_norm, norm(folder)).ratio()

        if score >= threshold:
            candidates.append({
                "rel_path": str(p.relative_to(vault_root)).replace("\\", "/"),
                "score": round(score, 3),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
