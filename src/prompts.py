SYSTEM_PROMPT = """You are an assistant that answers ONLY using the provided SOURCES.
Rules:
- Use ONLY facts found in SOURCES. If not found, say: "I couldn't find that in your notes."
- Do NOT invent details.
- Always cite sources in the form (source: <rel_path> [<heading>]).
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
