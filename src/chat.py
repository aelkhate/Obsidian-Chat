import argparse
from rich import print

from .local_llm import LocalLLM
from .orchestrator import run_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", dest="question", required=False)
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    llm = LocalLLM()

    out = run_query(question, llm)

    print(f"\n[bold]Intent:[/bold] {out['intent']}")
    print("\n[bold]Answer:[/bold]\n")
    print(out["answer"])

    print("\n[bold]Sources used:[/bold]")
    for i, s in enumerate(out["sources"], 1):
        if "heading" in s:
            print(f"{i}. {s.get('rel_path')} [{s.get('heading')}] score={s.get('score', 0):.4f}")
        else:
            print(f"{i}. {s.get('rel_path')} score={s.get('score', 0):.4f}")


if __name__ == "__main__":
    main()
