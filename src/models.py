from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DocChunk:
    chunk_id: str
    rel_path: str
    title: str
    heading: str
    text: str
    file_mtime: float
    file_sha256: str
    meta: Dict[str, Any]
