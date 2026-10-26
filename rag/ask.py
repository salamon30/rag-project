"""CLI: python -m rag.ask "your question" [--k 5] [--show-context]"""
import argparse

from dotenv import load_dotenv

load_dotenv()

from .pipeline import RagPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    r = RagPipeline().ask(args.question, args.k)
    print(f"\nQ: {r.question}\n\nA: {r.answer}\n\nSources:")
    for i, h in enumerate(r.hits, start=1):
        print(f"  [{i}] {h.source} p.{h.page}  (sim {h.score:.3f})")
        if args.show_context:
            print("      " + h.text[:300].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
