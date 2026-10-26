"""Stage C: python -m eval.run_eval [--k 5]

Retrieval metrics: source hit@1, hit@k, page hit@k, MRR@10.
Answer metrics (LLM judge, 1-5): faithfulness, correctness.
Writes eval/results.csv and eval/report.md.
"""
import argparse
import csv
import json
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingest import config  # noqa: E402
from rag.generator import Generator, build_context  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from .judge import Judge  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
MRR_DEPTH = 10


def with_retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def evaluate(k: int):
    tests = [json.loads(l) for l in (EVAL_DIR / "testset.jsonl").read_text().splitlines() if l.strip()]
    retriever, gen = Retriever(), Generator()
    judge = Judge(gen)
    rows = []
    for t in tests:
        hits = retriever.search(t["question"], max(k, MRR_DEPTH))
        src_ranks = [i for i, h in enumerate(hits, 1) if h.source == t["expected_source"]]
        first = src_ranks[0] if src_ranks else None
        top = hits[:k]
        page_hit = any(h.source == t["expected_source"] and h.page in t["expected_pages"] for h in top)

        try:
            answer = with_retry(lambda: gen.answer(t["question"], top))
            j = judge.score(t["question"], t["reference_answer"], build_context(top), answer)
            if _is_refusal(answer) and j["correctness"] is not None:
                # every test question is answerable, so a refusal is never correct;
                # enforced in code because the judge sometimes rated refusals 5
                j["correctness"] = 1
                j["correctness_reason"] = "Refusal on an answerable question (rule-based score)."
        except Exception as e:  # one bad call should not lose the whole run
            answer = f"GENERATION ERROR: {e}"
            j = {"faithfulness": None, "correctness": None,
                 "faithfulness_reason": "not judged: generation failed", "correctness_reason": ""}
        rows.append({
            "id": t["id"], "question": t["question"], "expected_source": t["expected_source"],
            "expected_pages": " ".join(map(str, t["expected_pages"])),
            "hit@1": int(first == 1), f"hit@{k}": int(first is not None and first <= k),
            f"page_hit@{k}": int(page_hit), "rr": round(1 / first, 3) if first else 0.0,
            "first_rank": first or "",
            "retrieved": "; ".join(f"{h.source}:p{h.page}" for h in top),
            "faithfulness": j["faithfulness"], "correctness": j["correctness"],
            "faithfulness_reason": j["faithfulness_reason"], "correctness_reason": j["correctness_reason"],
            "answer": answer, "reference_answer": t["reference_answer"],
        })
        print(f"{t['id']}  rank={first or '-':>2}  F={j['faithfulness']}  C={j['correctness']}")
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# Phrases about missing information only. A bare "does not include" matched a correct
# answer ("CI does not include zero"), so cited answers are never treated as refusals.
REFUSAL = re.compile(
    r"I don.t know|don.t see (any |specific )?information"
    r"|(documents?|context|passages?|text) (do|does) not (contain|provide|include|mention)"
    r"|no (specific |relevant )?information (about|on|regarding)", re.I)


def _is_refusal(answer):
    return bool(REFUSAL.search(answer[:300])) and not re.search(r"\[\d+\]", answer)


def summarize(rows, k):
    return {
        "questions": len(rows),
        "hit@1": mean([r["hit@1"] for r in rows]),
        f"hit@{k}": mean([r[f"hit@{k}"] for r in rows]),
        f"page_hit@{k}": mean([r[f"page_hit@{k}"] for r in rows]),
        f"MRR@{MRR_DEPTH}": mean([r["rr"] for r in rows]),
        "answers with [n] citations": mean([int(bool(re.search(r"\[\d+\]", r["answer"])))
                                            for r in rows if not _is_refusal(r["answer"])]),
        "refusals": sum(_is_refusal(r["answer"]) for r in rows),
        "faithfulness (1-5)": mean([r["faithfulness"] for r in rows]),
        "correctness (1-5)": mean([r["correctness"] for r in rows]),
        "unscored (errors)": sum(r["faithfulness"] is None for r in rows),
    }


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_report(rows, summary, k):
    with open(EVAL_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def score(r):
        f, c = r["faithfulness"] or 0, r["correctness"] or 0
        return (r[f"hit@{k}"], min(f, c), f + c, r["rr"])
    failures = [r for r in sorted(rows, key=score)
                if not r[f"hit@{k}"] or (r["faithfulness"] or 0) <= 3 or (r["correctness"] or 0) <= 3][:3]
    if not failures:  # still show the weakest cases so the report is informative
        failures = sorted(rows, key=score)[:3]
        fail_title = "No hard failures; weakest cases"
    else:
        fail_title = "Example failures"

    L = [f"# RAG evaluation report", "",
         f"Date: {date.today()}  ",
         f"Embeddings: `{config.EMBED_MODEL}` · Generator and judge: `{config.LLM_MODEL}` "
         f"(generator thinking {'on' if config.LLM_THINKING else 'off'} with no-thinking fallback, judge thinking off) · k = {k}", "",
         "## Summary", "", "| Metric | Value |", "|---|---|"]
    L += [f"| {m} | {fmt(v)} |" for m, v in summary.items()]
    L += ["", "**Metric definitions.** hit@k: expected source file is among the top-k chunks. "
          "page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. "
          f"MRR@{MRR_DEPTH}: mean of 1/rank of the first chunk from the expected file (0 if not in top {MRR_DEPTH}). "
          "Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. "
          "Both are scored 1-5 by an LLM judge.", "",
          "**Caveat.** The judge is the same model as the generator, which can bias scores upward "
          "(self-preference). Treat answer scores as indicative, not absolute.", "",
          "## Per-question results", "",
          f"| id | question | first rank | hit@{k} | page hit | F | C |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r["question"].replace("|", "/")
        L.append(f"| {r['id']} | {q} | {r['first_rank'] or '–'} | {r[f'hit@{k}']} | "
                 f"{r[f'page_hit@{k}']} | {r['faithfulness']} | {r['correctness']} |")
    L += ["", f"## {fail_title}", ""]
    for r in failures:
        L += [f"### {r['id']}: {r['question']}", "",
              f"- **Expected:** `{r['expected_source']}` p.{r['expected_pages']}",
              f"- **Retrieved:** {r['retrieved']}",
              f"- **Answer:** {r['answer']}",
              f"- **Reference:** {r['reference_answer']}",
              f"- **Faithfulness {r['faithfulness']}:** {r['faithfulness_reason']}",
              f"- **Correctness {r['correctness']}:** {r['correctness_reason']}", ""]
    (EVAL_DIR / "report.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    k = ap.parse_args().k
    rows = evaluate(k)
    summary = summarize(rows, k)
    write_report(rows, summary, k)
    print("\n| Metric | Value |\n|---|---|")
    for m, v in summary.items():
        print(f"| {m} | {fmt(v)} |")
    print(f"\nWrote {EVAL_DIR / 'report.md'} and results.csv")


if __name__ == "__main__":
    main()
