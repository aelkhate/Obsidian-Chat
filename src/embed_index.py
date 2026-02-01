import os
import json
from typing import List, Dict, Any
import datetime
import numpy as np
import faiss
from rich import print
from sentence_transformers import SentenceTransformer

from .config import VAULT_PATH
from .ingest import load_and_chunk_vault

INDEX_DIR = "data/index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

# A good default embedding model (fast + solid)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def _json_default(o):
    # Convert dates/times to ISO 8601 strings
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    # Fallback: stringify anything else non-serializable
    return str(o)

def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")



def build_index():
    ensure_dir(INDEX_DIR)

    print(f"[bold]Vault:[/bold] {VAULT_PATH}")
    chunks = load_and_chunk_vault(VAULT_PATH)
    print(f"[green]Chunks to index:[/green] {len(chunks)}")

    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c.text for c in chunks]
    print("[bold]Embedding chunks...[/bold]")

    # SentenceTransformers returns float32 numpy array by default if convert_to_numpy=True
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity with inner product
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"[bold]Embedding dim:[/bold] {dim}")

    # Use Inner Product index; with normalized embeddings, IP ~= cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    print(f"[green]Saved FAISS index:[/green] {FAISS_PATH}")

    # Save metadata aligned with FAISS row ids
    meta_rows = []
    for i, c in enumerate(chunks):
        meta_rows.append(
            {
                "row_id": i,
                "chunk_id": c.chunk_id,
                "rel_path": c.rel_path,
                "title": c.title,
                "heading": c.heading,
                "text": c.text,
                "file_mtime": c.file_mtime,
                "file_sha256": c.file_sha256,
                "frontmatter": c.meta,
            }
        )

    save_jsonl(META_PATH, meta_rows)
    print(f"[green]Saved chunk metadata:[/green] {META_PATH}")

    # Minimal manifest for reproducibility
    manifest = {
        "vault_path": VAULT_PATH,
        "embedding_model": EMBED_MODEL_NAME,
        "num_chunks": len(chunks),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[green]Saved manifest:[/green] {MANIFEST_PATH}")
    print("[bold]Index build complete.[/bold]")


if __name__ == "__main__":
    build_index()
