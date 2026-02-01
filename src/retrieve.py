import os
import json
from typing import List, Dict, Any, Optional

import faiss
from sentence_transformers import SentenceTransformer
from rich import print

INDEX_DIR = "data/index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")


def load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata() -> List[Dict[str, Any]]:
    rows = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def retrieve(
    query: str,
    top_k: int = 5,
    path_prefix: Optional[str] = None,
    overfetch: int = 200,
) -> List[Dict[str, Any]]:
    """
    Vector retrieval with OPTIONAL folder filtering.

    If path_prefix is provided (e.g. "03_projects"):
    1) Overfetch from FAISS (because FAISS can't filter)
    2) Filter by rel_path.startswith(path_prefix)
    3) Return top_k filtered results
    """
    manifest = load_manifest()
    embed_model_name = manifest["embedding_model"]

    model = SentenceTransformer(embed_model_name)
    index = faiss.read_index(FAISS_PATH)
    meta = load_metadata()

    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = max(overfetch, top_k) if path_prefix else top_k
    scores, ids = index.search(q_vec, k)

    prefix_norm = None
    if path_prefix:
        prefix_norm = path_prefix.rstrip("/") + "/"

    results = []
    for score, rid in zip(scores[0], ids[0]):
        if rid < 0:
            continue

        row = meta[rid]

        if prefix_norm and not row["rel_path"].startswith(prefix_norm):
            continue

        results.append(
            {
                "score": float(score),
                "chunk_id": row["chunk_id"],
                "rel_path": row["rel_path"],
                "title": row["title"],
                "heading": row["heading"],
                "text": row["text"],
            }
        )

        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    q = input("Query: ").strip()
    prefix = input("Path prefix filter (optional, e.g. 03_projects): ").strip()
    prefix = prefix if prefix else None

    out = retrieve(q, top_k=5, path_prefix=prefix, overfetch=300 if prefix else 0)

    print("\n[bold]Top results:[/bold]")
    for i, r in enumerate(out, 1):
        print(f"\n#{i} score={r['score']:.4f}  {r['rel_path']}  [{r['heading']}]")
        print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
