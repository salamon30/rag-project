"""Second opinion on answer scores: python -m eval.rejudge --model MODEL [--sets "" _hard _heldout]

The main evaluation uses the generator's own model as judge, which can inflate scores
(self-preference). This script re-scores the *saved* answers with a different model.
No answer is regenerated: the passages are fetched again with the saved search queries,
so both judges see the same question, context and answer.

Writes eval/rejudge{suffix}.csv per test set and eval/judge_comparison.md.
"""
import argparse
import csv
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from ingest import config  # noqa: E402
from rag.generator import Generator, build_context  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

from .judge import Judge  # noqa: E402
from .run_eval import EVAL_DIR, MRR_DEPTH, apply_refusal_rules, fmt, load_results, mean  # noqa: E402

SET_NAMES = {"": "Original", "_hard": "Hard", "_heldout": "Held-out"}


def as_test(row):
    """A saved row has no expected_pages dict; the refusal rules only need answerability."""
    return {**row, "expected_pages": {} if row["type"] == "unanswerable" else {"_": []}}


def replay_context(retriever, row, k):
    """Fetch the passages again with the saved queries. Returns (context, same_pages)."""
    queries = [q for q in row["queries"].split(" || ") if q] or [row["question"]]
    top = retriever.search_many(queries, max(k, MRR_DEPTH), question=row["question"])[:k]
    pages = "; ".join(f"{h.source}:p{h.page}" for h in top)
    return build_context(top), pages == row["retrieved"]


def rejudge(rows, retriever, judge, k, keep=None):
    """Score saved answers with the second judge. `keep` maps id -> an earlier second-judge
    row; rows that already have a score are reused, so a retry only repeats failed calls."""
    out = []
    for r in rows:
        if r["correctness"] is None or r["answer"].startswith("GENERATION ERROR"):
            continue
        if keep and keep.get(r["id"], {}).get("correctness_second") is not None:
            out.append(keep[r["id"]])
            continue
        context, same = replay_context(retriever, r, k)
        j = judge.score(r["question"], r["reference_answer"], context, r["answer"])
        apply_refusal_rules(as_test(r), r["answer"], j)
        out.append({
            "id": r["id"], "type": r["type"], "context_matches": int(same),
            "correctness_first": r["correctness"], "correctness_second": j["correctness"],
            # faithfulness is only comparable when the second judge saw the same passages
            "faithfulness_first": r["faithfulness"], "faithfulness_second": j["faithfulness"] if same else None,
            "correctness_reason_second": j["correctness_reason"],
            "faithfulness_reason_second": j["faithfulness_reason"],
        })
        print(f"{r['id']} {r['type']:<13} C {r['correctness']} -> {j['correctness']}   "
              f"F {r['faithfulness']} -> {j['faithfulness']}{'' if same else '   (context differs)'}")
    return out


def agreement(rows, metric="correctness"):
    """Mean scores of both judges and how often they agree (exactly / within one point)."""
    pairs = [(r[f"{metric}_first"], r[f"{metric}_second"]) for r in rows
             if r[f"{metric}_first"] is not None and r[f"{metric}_second"] is not None]
    if not pairs:
        return {"n": 0, "first": float("nan"), "second": float("nan"), "exact": float("nan"),
                "within_one": float("nan")}
    return {
        "n": len(pairs),
        "first": mean([a for a, _ in pairs]),
        "second": mean([b for _, b in pairs]),
        "exact": mean([int(a == b) for a, b in pairs]),
        "within_one": mean([int(abs(a - b) <= 1) for a, b in pairs]),
    }


def load_rejudged(suffix):
    """Earlier second-judge rows by id, or {} if this set was never re-judged."""
    path = EVAL_DIR / f"rejudge{suffix}.csv"
    if not path.exists():
        return {}
    ints = {"context_matches", "correctness_first", "correctness_second", "faithfulness_first", "faithfulness_second"}
    with open(path, newline="") as f:
        return {row["id"]: {c: (int(v) if v else None) if c in ints else v for c, v in row.items()}
                for row in csv.DictReader(f)}


def write_csv(rows, suffix):
    with open(EVAL_DIR / f"rejudge{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_comparison(results, first_model, second_model):
    lines = [
        "# Judge comparison", "",
        f"Date: {date.today()}  ",
        f"First judge (same model as the generator): `{first_model}`  ",
        f"Second judge (independent): `{second_model}`", "",
        "The saved answers were scored again by a model from a different vendor. Nothing was "
        "regenerated: both judges saw the same question, passages, answer and reference. "
        "Refusals are scored by the same code rules for both.", "",
        "## Correctness (1-5)", "",
        "| Test set | Answers | First judge | Second judge | Same score | Within 1 point |",
        "|---|---|---|---|---|---|",
    ]
    for suffix, rows in results.items():
        a = agreement(rows)
        lines.append(f"| {SET_NAMES.get(suffix, suffix)} | {a['n']} | {fmt(a['first'])} | {fmt(a['second'])} "
                     f"| {a['exact']:.0%} | {a['within_one']:.0%} |")
    lines += ["", "## Faithfulness (1-5)", "",
              "| Test set | Answers | First judge | Second judge | Same score | Within 1 point |",
              "|---|---|---|---|---|---|"]
    for suffix, rows in results.items():
        a = agreement(rows, "faithfulness")
        lines.append(f"| {SET_NAMES.get(suffix, suffix)} | {a['n']} | {fmt(a['first'])} | {fmt(a['second'])} "
                     f"| {a['exact']:.0%} | {a['within_one']:.0%} |")
    lines += ["", "## Where the judges disagree on correctness", "",
              "| Set | Id | Type | First | Second | Second judge's reason |", "|---|---|---|---|---|---|"]
    n = 0
    for suffix, rows in results.items():
        for r in rows:
            a, b = r["correctness_first"], r["correctness_second"]
            if a is not None and b is not None and a != b:
                n += 1
                reason = (r["correctness_reason_second"] or "").replace("|", "/").replace("\n", " ")
                lines.append(f"| {SET_NAMES.get(suffix, suffix)} | {r['id']} | {r['type']} | {a} | {b} | {reason} |")
    if not n:
        lines.append("| – | – | – | – | – | no disagreements |")
    (EVAL_DIR / "judge_comparison.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="second judge, e.g. minimaxai/minimax-m3.1-flash")
    ap.add_argument("--sets", nargs="*", default=["", "_hard", "_heldout"],
                    help="results file suffixes: '' (original), _hard, _heldout")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--retry-missing", action="store_true",
                    help="keep the scores in eval/rejudge*.csv and only repeat calls that failed")
    args = ap.parse_args()

    gen = Generator()
    retriever, judge = Retriever(), Judge(gen, model=args.model)
    results = {}
    for suffix in args.sets:
        print(f"\n== {SET_NAMES.get(suffix, suffix)} ==")
        keep = load_rejudged(suffix) if args.retry_missing else None
        results[suffix] = rejudge(load_results(suffix), retriever, judge, args.k, keep)
        write_csv(results[suffix], suffix)
    write_comparison(results, gen.model, args.model)
    print(f"\nWrote {EVAL_DIR / 'judge_comparison.md'}")


if __name__ == "__main__":
    main()
