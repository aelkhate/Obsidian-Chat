import argparse
from rich import print

from .retrieve import retrieve
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .local_llm import LocalLLM


def format_sources(results, max_chars_per_source=900) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        text = r["text"].strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "..."
        blocks.append(
            f"[{i}] rel_path: {r['rel_path']}\n"
            f"    heading: {r['heading']}\n"
            f"    chunk_id: {r['chunk_id']}\n"
            f"    score: {r['score']:.4f}\n"
            f"    text:\n{text}\n"
        )
    return "\n".join(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", dest="question", required=False)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--overfetch", type=int, default=300)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    path_prefix = args.path.strip() or None

    results = retrieve(
        question,
        top_k=args.k,
        path_prefix=path_prefix,
        overfetch=args.overfetch,
    )

    if not results:
        print("[yellow]No sources retrieved.[/yellow]")
        return

    sources_block = format_sources(results)
    user_prompt = build_user_prompt(question, sources_block)

    llm = LocalLLM()
    answer = llm.generate(SYSTEM_PROMPT, user_prompt, max_new_tokens=args.max_new_tokens)

    print("\n[bold]Answer:[/bold]\n")
    print(answer)

    print("\n[bold]Sources used (retrieved):[/bold]")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['rel_path']} [{r['heading']}] score={r['score']:.4f}")


if __name__ == "__main__":
    main()
