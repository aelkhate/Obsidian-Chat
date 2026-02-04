# src/chat.py
import argparse
from rich import print
from rich.panel import Panel
from rich.json import JSON

from .local_llm import LocalLLM
from .orchestrator import run_query, TRACE_PATH_DEFAULT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", dest="question", required=False)
    parser.add_argument("--debug", action="store_true", help="Print plan + verifier details")
    parser.add_argument("--no-trace", action="store_true", help="Disable JSONL tracing")
    parser.add_argument("--trace-path", default=TRACE_PATH_DEFAULT, help="Path to traces.jsonl")
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    llm = LocalLLM()

    out = run_query(
        question,
        llm,
        trace_path=args.trace_path,
        enable_trace=not args.no_trace,
    )

    print(f"\n[bold]Intent:[/bold] {out.get('intent')}")

    verifier = out.get("verifier")
    if verifier:
        print(f"[bold]Verifier:[/bold] {verifier.get('status')}")
    else:
        print("[bold]Verifier:[/bold] (not run)")

    print("\n[bold]Answer:[/bold]\n")
    print(out.get("answer", ""))

    if out.get("repaired"):
        print("\n[yellow]Note:[/yellow] Answer was rewritten once to improve citation compliance (observer mode).")

    if args.debug:
        if "plan" in out:
            print(Panel(JSON.from_data(out["plan"]), title="Plan JSON", expand=False))
        if "verifier_before_repair" in out:
            print(Panel(JSON.from_data(out["verifier_before_repair"]), title="Verifier BEFORE repair", expand=False))
        if "verifier" in out:
            print(Panel(JSON.from_data(out["verifier"]), title="Verifier Verdict", expand=False))

    print("\n[bold]Sources used:[/bold]")
    srcs = out.get("sources", []) or []
    if not srcs:
        print("(none)")
        return

    for i, s in enumerate(srcs, 1):
        heading = s.get("heading")
        score = s.get("score", 0.0)
        if heading is not None:
            print(f"{i}. {s.get('rel_path')} [{heading}] score={score:.4f}")
        else:
            print(f"{i}. {s.get('rel_path')} score={score:.4f}")


if __name__ == "__main__":
    main()
