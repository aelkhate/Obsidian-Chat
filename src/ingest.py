import os
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Iterable, Tuple
from .models import DocChunk

import frontmatter
from rich import print

from .config import VAULT_PATH


# -------------------------
# File iteration
# -------------------------
EXCLUDE_DIRS = {
    ".obsidian",
    ".git",
    ".trash",
    "98_attachements",
    ".makemd",
}

def iter_md_files(vault_dir: str):
    for root, dirs, files in os.walk(vault_dir):
        # remove excluded dirs from traversal
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            lf = f.lower()
            if lf.endswith(".md") or lf.endswith(".markdown"):
                yield os.path.join(root, f)

# -------------------------
# Markdown splitting + chunking
# -------------------------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


def split_by_headings(md_text: str) -> List[Tuple[str, str]]:
    """
    Split markdown text into (heading, body) sections based on Markdown headings.
    If there is text before the first heading, it's stored under heading "ROOT".
    """
    sections: List[Tuple[str, str]] = []
    current_heading = "ROOT"
    buf: List[str] = []

    def flush():
        nonlocal buf
        body = "\n".join(buf).strip()
        if body:
            sections.append((current_heading, body))
        buf = []

    for line in md_text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            current_heading = m.group(2).strip() or "UNTITLED"
        else:
            buf.append(line)

    flush()
    return sections


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    """
    Chunk a section body by character length with overlap.
    This is simple and robust; later we can upgrade to token-based chunking.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


# -------------------------
# Hashing
# -------------------------
def sha256_of_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


# -------------------------
# Main ingestion
# -------------------------
def load_and_chunk_vault(vault_dir: str) -> List[DocChunk]:
    chunks: List[DocChunk] = []

    # audit counters
    md_files = 0
    nonempty_files = 0
    chunked_files = 0
    parse_failed = 0

    for abs_path in iter_md_files(vault_dir):
        md_files += 1

        rel_path = os.path.relpath(abs_path, vault_dir)
        title = os.path.splitext(os.path.basename(abs_path))[0]

        # read file
        raw = open(abs_path, "r", encoding="utf-8", errors="ignore").read()
        if raw.strip():
            nonempty_files += 1

        # file stats
        stat = os.stat(abs_path)
        mtime = stat.st_mtime

        # parse frontmatter safely
        try:
            post = frontmatter.loads(raw)
            meta = dict(post.metadata) if post.metadata else {}
            content = post.content if post.content is not None else ""
        except Exception as e:
            parse_failed += 1
            meta = {}
            content = raw  # treat as plain markdown
            print(f"[yellow]Warning:[/yellow] Frontmatter parse failed in {rel_path}: {e}")

        # hash only the content (not metadata) to detect meaningful content changes
        file_hash = sha256_of_text(content)

        # split into sections
        sections = split_by_headings(content)

        # fallback: if splitting yielded nothing but file has content
        if not sections and content.strip():
            sections = [("ROOT", content.strip())]

        produced = 0

        # create chunks
        for si, (heading, body) in enumerate(sections):
            for ci, ctext in enumerate(chunk_text(body)):
                chunk_id = f"{rel_path}::s{si}::c{ci}"
                chunks.append(
                    DocChunk(
                        chunk_id=chunk_id,
                        rel_path=rel_path,
                        title=title,
                        heading=heading,
                        text=ctext,
                        file_mtime=mtime,
                        file_sha256=file_hash,
                        meta=meta,
                    )
                )
                produced += 1

        if produced > 0:
            chunked_files += 1

    print(f"[bold]Scanned .md files:[/bold] {md_files}")
    print(f"[bold]Non-empty .md files:[/bold] {nonempty_files}")
    print(f"[bold]Files that produced chunks:[/bold] {chunked_files}")
    print(f"[bold]Frontmatter parse failed:[/bold] {parse_failed}")

    return chunks


# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    print(f"[bold]Vault:[/bold] {VAULT_PATH}")
    docs = load_and_chunk_vault(VAULT_PATH)
    print(f"[green]Loaded chunks:[/green] {len(docs)}")

    if docs:
        sample = docs[0]
        preview = sample.text[:200] + ("..." if len(sample.text) > 200 else "")
        print("[bold]Sample chunk:[/bold]")
        print(
            {
                "chunk_id": sample.chunk_id,
                "rel_path": sample.rel_path,
                "title": sample.title,
                "heading": sample.heading,
                "text_preview": preview,
            }
        )
