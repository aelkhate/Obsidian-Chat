import os
import json
from typing import Dict, List, Tuple

INDEX_DIR = "data/index"
META_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")


def build_note_catalog() -> List[Dict]:
    """
    Builds a catalog of unique notes from the chunk metadata:
    - rel_path
    - title
    - headings (unique headings encountered)
    """
    notes: Dict[str, Dict] = {}

    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rel = r["rel_path"]
            if rel not in notes:
                notes[rel] = {
                    "rel_path": rel,
                    "title": r.get("title", os.path.splitext(os.path.basename(rel))[0]),
                    "headings": set(),
                }
            notes[rel]["headings"].add(r.get("heading", "ROOT"))

    out = []
    for rel, row in notes.items():
        row["headings"] = sorted(list(row["headings"]))
        out.append(row)

    return out
