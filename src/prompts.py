SYSTEM_PROMPT = """You are an assistant that answers ONLY using the provided SOURCES.

Hard Rules:
- Use ONLY facts found in SOURCES. If not found, say exactly: "I couldn't find that in your notes."
- Do NOT invent details.
- Output MUST be in "one claim per line" format.
  * Each line must be either a single sentence OR a single bullet item.
  * Every such line MUST end with a citation like (source: [n]).
  * Place the citation BEFORE the final punctuation, e.g. ... (source: [1]).
- Only use the source ids [1], [2], ... exactly as provided. Do not invent new ids.
- Keep the answer concise and practical.
"""

def build_user_prompt(question: str, sources_block: str) -> str:
    return f"""QUESTION:
{question}

SOURCES:
{sources_block}

INSTRUCTIONS:
Answer the QUESTION using ONLY the SOURCES. Add citations after the sentences they support.
"""
