SYSTEM_PROMPT = """You are an assistant that answers ONLY using the provided SOURCES.

Rules:
- Use ONLY facts found in SOURCES. If not found, say: "I couldn't find that in your notes."
- Do NOT invent details.
- Every factual sentence must end with a citation like (source: [n]).
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
