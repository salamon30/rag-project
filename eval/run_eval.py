"""Stage C: python -m eval.run_eval [--k 5] [--testset testset_hard]

Retrieval metrics (answerable questions): source hit@1, hit@k, all_sources@k, page hit@k, MRR@10.
Answer metrics (LLM judge, 1-5): faithfulness, correctness.
Refusal metrics: refusal recall on unanswerable questions, false refusals on answerable ones.
Writes eval/results{suffix}.csv and eval/report{suffix}.md, e.g. report_hard.md for testset_hard.

Question types: single (one paper, wording close to the paper), unanswerable (topic is in
the corpus but the fact is not), false_premise (the question assumes something wrong),
multi_hop (needs two papers), paraphrase (avoids the paper's keywords).
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
from rag.decompose import Decomposer  # noqa: E402
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


def load_tests(name="testset"):
    """Load a test set and normalise it: expected_pages becomes {source: [pages]}.

    The original set has one expected_source and a page list; newer sets give
    expected_pages per source so a question can need several papers."""
    tests = [json.loads(line) for line in (EVAL_DIR / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    for t in tests:
        t.setdefault("type", "single")
        if "expected_source" in t:
            t["expected_pages"] = {t.pop("expected_source"): t["expected_pages"]}
    return tests


def answerable(t):
    return t["type"] != "unanswerable"


def rank_metrics(hits, expected, k):
    """Retrieval metrics for one question. expected = {source: [pages]}."""
    ranks = [i for i, h in enumerate(hits, 1) if h.source in expected]
    first = ranks[0] if ranks else None
    top = hits[:k]
    return {
        "hit@1": int(first == 1),
        f"hit@{k}": int(first is not None and first <= k),
        # every expected paper is in the top-k (only differs from hit@k for multi_hop)
        f"all_sources@{k}": int(all(any(h.source == s for h in top) for s in expected)),
        f"page_hit@{k}": int(any(h.page in expected.get(h.source, ()) for h in top)),
        "rr": 1 / first if first else 0.0,
        "first_rank": first,
    }


def search(retriever, decomposer, question, n):
    """Run the question's searches. Returns the hits and the queries that were used."""
    queries = decomposer.split(question) if decomposer else [question]
    hits = retriever.search_many(queries, n, question=question)
    return hits, getattr(retriever, "last_searches", queries)


def retrieval_metrics(retriever, tests, k, decomposer=None):
    """Retrieval only. No LLM calls unless a decomposer is given (one short call per question)."""
    per_q = {t["id"]: rank_metrics(search(retriever, decomposer, t["question"], max(k, MRR_DEPTH))[0],
                                   t["expected_pages"], k)
             for t in tests if answerable(t)}
    names = ["hit@1", f"hit@{k}", f"all_sources@{k}", f"page_hit@{k}"]
    summary = {m: mean([r[m] for r in per_q.values()]) for m in names}
    summary[f"MRR@{MRR_DEPTH}"] = mean([r["rr"] for r in per_q.values()])
    misses = [qid for qid, r in per_q.items() if not r[f"page_hit@{k}"]]
    return summary, misses


def apply_refusal_rules(t, answer, j):
    """Score refusals in code: the judge sometimes rated an answerable refusal 5.

    Answerable question + refusal -> correctness 1.
    Unanswerable question -> correctness 5 if the answer declines, else 1 (it made something up).
    Returns whether the answer said it does not know."""
    if answerable(t):
        said_dont_know = _is_refusal(answer)
        if said_dont_know and j["correctness"] is not None:
            j["correctness"], j["correctness_reason"] = 1, "Refusal on an answerable question (rule-based score)."
    else:
        # cited partial answers are fine here ("the papers give hardware but no cost [2]")
        said_dont_know = _declines(answer)
        if j["correctness"] is not None:
            j["correctness"] = 5 if said_dont_know else 1
            j["correctness_reason"] = ("Correctly declined an unanswerable question (rule-based score)."
                                       if said_dont_know else
                                       "Answered a question the documents cannot answer (rule-based score).")
    return said_dont_know


def evaluate(k: int, rerank_model: str, testset: str, decompose: bool = False):
    tests = load_tests(testset)
    retriever, gen = Retriever(rerank_model=rerank_model), Generator()
    judge = Judge(gen)
    decomposer = Decomposer(gen) if decompose else None
    rows = []
    for t in tests:
        hits, queries = search(retriever, decomposer, t["question"], max(k, MRR_DEPTH))
        top = hits[:k]
        m = rank_metrics(hits, t["expected_pages"], k) if answerable(t) else dict.fromkeys(
            ["hit@1", f"hit@{k}", f"all_sources@{k}", f"page_hit@{k}", "rr", "first_rank"])

        try:
            answer = with_retry(lambda t=t, top=top: gen.answer(t["question"], top))
            j = judge.score(t["question"], t["reference_answer"], build_context(top), answer)
            said_dont_know = apply_refusal_rules(t, answer, j)
        except Exception as e:  # one bad call should not lose the whole run
            answer, said_dont_know = f"GENERATION ERROR: {e}", None
            j = {"faithfulness": None, "correctness": None,
                 "faithfulness_reason": "not judged: generation failed", "correctness_reason": ""}
        rows.append({
            "id": t["id"], "type": t["type"], "question": t["question"], "queries": " || ".join(queries),
            "expected": "; ".join(f"{s}:p{' '.join(map(str, p))}" for s, p in t["expected_pages"].items()),
            **{key: m[key] for key in ("hit@1", f"hit@{k}", f"all_sources@{k}", f"page_hit@{k}")},
            "rr": round(m["rr"], 3) if m["rr"] is not None else None,
            "first_rank": m["first_rank"] or "",
            "retrieved": "; ".join(f"{h.source}:p{h.page}" for h in top),
            "said_dont_know": None if said_dont_know is None else int(said_dont_know),
            "faithfulness": j["faithfulness"], "correctness": j["correctness"],
            "faithfulness_reason": j["faithfulness_reason"], "correctness_reason": j["correctness_reason"],
            "answer": answer, "reference_answer": t["reference_answer"],
        })
        print(f"{t['id']} {t['type']:<13} rank={m['first_rank'] or '-':>2}  F={j['faithfulness']}  C={j['correctness']}"
              f"  searches={len(queries)}")
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


# Looser check for unanswerable questions: the answer may cite what *is* there
# while saying the requested fact is missing.
DECLINE = re.compile(
    REFUSAL.pattern + r"|(not|n.t) (specified|stated|reported|mentioned|provided|given|available|disclosed)"
    r"|(do|does) not (specify|state|report|give|disclose)|no (mention|information|data) (of|about|on)"
    r"|\bno\b[^.]{0,80}\b(is|are|was|were) (reported|provided|mentioned|described|given|stated)", re.I)


def _is_refusal(answer):
    return bool(REFUSAL.search(answer[:300])) and not re.search(r"\[\d+\]", answer)


def _declines(answer):
    return bool(DECLINE.search(answer))


def summarize(rows, k):
    ans = [r for r in rows if r["type"] != "unanswerable"]
    unans = [r for r in rows if r["type"] == "unanswerable"]
    s = {
        "questions": len(rows),
        "hit@1": mean([r["hit@1"] for r in ans]),
        f"hit@{k}": mean([r[f"hit@{k}"] for r in ans]),
    }
    if any(r["type"] == "multi_hop" for r in rows):
        s[f"all_sources@{k}"] = mean([r[f"all_sources@{k}"] for r in ans])
    s |= {
        f"page_hit@{k}": mean([r[f"page_hit@{k}"] for r in ans]),
        f"MRR@{MRR_DEPTH}": mean([r["rr"] for r in ans]),
        "answers with [n] citations": mean([int(bool(re.search(r"\[\d+\]", r["answer"])))
                                            for r in ans if not r["said_dont_know"]]),
        "refusals": sum(bool(r["said_dont_know"]) for r in ans),
    }
    if unans:
        s["refusal recall (unanswerable)"] = mean([r["said_dont_know"] for r in unans])
    s |= {
        "faithfulness (1-5)": mean([r["faithfulness"] for r in rows]),
        "correctness (1-5)": mean([r["correctness"] for r in rows]),
        "unscored (errors)": sum(r["faithfulness"] is None for r in rows),
    }
    return s


def by_type(rows, k):
    """Per question type: count, page hit, correctness, how often it said 'I don't know'."""
    out = {}
    for typ in dict.fromkeys(r["type"] for r in rows):
        rs = [r for r in rows if r["type"] == typ]
        out[typ] = {"n": len(rs), f"page_hit@{k}": mean([r[f"page_hit@{k}"] for r in rs]),
                    "correctness": mean([r["correctness"] for r in rs]),
                    "dont_know": sum(bool(r["said_dont_know"]) for r in rs)}
    return out


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_report(rows, summary, k, suffix=""):
    with open(EVAL_DIR / f"results{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def score(r):
        f, c = r["faithfulness"] or 0, r["correctness"] or 0
        hit = 1 if r[f"hit@{k}"] is None else r[f"hit@{k}"]  # unanswerable: nothing to retrieve
        return (min(f, c), hit, f + c, r["rr"] or 0)
    failures = [r for r in sorted(rows, key=score)
                if r[f"hit@{k}"] == 0 or (r["faithfulness"] or 0) <= 3 or (r["correctness"] or 0) <= 3][:5]
    if not failures:  # still show the weakest cases so the report is informative
        failures = sorted(rows, key=score)[:3]
        fail_title = "No hard failures; weakest cases"
    else:
        fail_title = "Example failures"

    L = ["# RAG evaluation report", "",
         f"Date: {date.today()}  ",
         f"Embeddings: `{config.EMBED_MODEL}` · Generator and judge: `{config.LLM_MODEL}` "
         f"(generator thinking {'on' if config.LLM_THINKING else 'off'} with no-thinking fallback, judge thinking off) · k = {k}  ",
         f"Reranker: `{RERANK}`" + (f" over top {config.RERANK_CANDIDATES} vector candidates" if RERANK.lower() != "none" else " (vector search only)")
         + (f" · at most {config.MAX_PER_SOURCE} chunks per paper" if config.MAX_PER_SOURCE else "")
         + (" · query decomposition on" if DECOMPOSE else ""), "",
         "## Summary", "", "| Metric | Value |", "|---|---|"]
    L += [f"| {m} | {fmt(v)} |" for m, v in summary.items()]
    L += ["", "**Metric definitions.** hit@k: expected source file is among the top-k chunks. "
          "page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. "
          f"MRR@{MRR_DEPTH}: mean of 1/rank of the first chunk from the expected file (0 if not in top {MRR_DEPTH}). "
          "Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. "
          "Both are scored 1-5 by an LLM judge.", "",
          "**Caveat.** The judge is the same model as the generator, which can bias scores upward "
          "(self-preference). Treat answer scores as indicative, not absolute.", "",
          ]
    types = by_type(rows, k)
    if len(types) > 1:
        L += ["## By question type", "",
              f"| type | n | page_hit@{k} | correctness (1-5) | said don't know |", "|---|---|---|---|---|"]
        for typ, v in types.items():
            ph = v[f"page_hit@{k}"]
            ph = "–" if ph != ph else fmt(ph)  # NaN for unanswerable: nothing to retrieve
            L.append(f"| {typ} | {v['n']} | {ph} | {fmt(v['correctness'])} | {v['dont_know']} |")
        L += ["", "Unanswerable questions should all be declined; every other type should be answered.", ""]
    L += ["## Per-question results", "",
          f"| id | type | question | first rank | hit@{k} | page hit | F | C |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r["question"].replace("|", "/")
        L.append(f"| {r['id']} | {r['type']} | {q} | {r['first_rank'] or '–'} | {'–' if r[f'hit@{k}'] is None else r[f'hit@{k}']} | "
                 f"{'–' if r[f'page_hit@{k}'] is None else r[f'page_hit@{k}']} | {r['faithfulness']} | {r['correctness']} |")
    L += ["", f"## {fail_title}", ""]
    for r in failures:
        L += [f"### {r['id']}: {r['question']}", "",
              f"- **Type:** {r['type']} · **Expected:** {r['expected'] or 'nothing (unanswerable)'}",
              f"- **Retrieved:** {r['retrieved']}",
              f"- **Answer:** {r['answer']}",
              f"- **Reference:** {r['reference_answer']}",
              f"- **Faithfulness {r['faithfulness']}:** {r['faithfulness_reason']}",
              f"- **Correctness {r['correctness']}:** {r['correctness_reason']}", ""]
    (EVAL_DIR / f"report{suffix}.md").write_text("\n".join(L))


def load_results(suffix):
    """Read a saved results CSV back into rows with numbers and None restored."""
    def num(v):
        if v == "":
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    text_cols = {"id", "type", "question", "queries", "expected", "retrieved", "answer", "reference_answer",
                 "faithfulness_reason", "correctness_reason"}
    with open(EVAL_DIR / f"results{suffix}.csv", newline="") as f:
        return [{c: (v if c in text_cols else num(v)) for c, v in row.items()} for row in csv.DictReader(f)]


def rescore(rows):
    """Re-apply the rule-based refusal scoring to saved answers (no LLM calls).

    Used when a scoring rule is fixed, so a measurement bug does not need a full rerun."""
    changed = []
    for r in rows:
        if r["correctness"] is None or r["answer"].startswith("GENERATION ERROR"):
            continue
        before = (r["said_dont_know"], r["correctness"])
        j = {"correctness": r["correctness"], "correctness_reason": r["correctness_reason"]}
        r["said_dont_know"] = int(apply_refusal_rules(r, r["answer"], j))
        r["correctness"], r["correctness_reason"] = j["correctness"], j["correctness_reason"]
        if (r["said_dont_know"], r["correctness"]) != before:
            changed.append(r["id"])
    return changed


RERANK = config.RERANK_MODEL
DECOMPOSE = config.DECOMPOSE


def main():
    global RERANK, DECOMPOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--rerank-model", default=config.RERANK_MODEL,
                    help="cross-encoder name, or 'none' for vector search only")
    ap.add_argument("--retrieval-only", nargs="*", metavar="MODEL",
                    help="compare retrieval metrics (no LLM) for 'none' and the given rerankers")
    ap.add_argument("--max-per-source", type=int, nargs="*", metavar="N",
                    help="with --retrieval-only: also compare these per-paper caps (0 = no cap)")
    ap.add_argument("--decompose", action=argparse.BooleanOptionalAction, default=config.DECOMPOSE,
                    help="split multi-paper questions into one search per paper (default from config)")
    ap.add_argument("--rescore", metavar="SUFFIX", nargs="?", const="",
                    help="re-apply scoring rules to eval/results{SUFFIX}.csv without LLM calls")
    ap.add_argument("--testset", default="testset",
                    help="file name in eval/ without .jsonl, e.g. testset_hard")
    args = ap.parse_args()
    k = args.k
    suffix = args.testset.removeprefix("testset")  # testset_hard -> report_hard.md

    if args.rescore is not None:
        rows = load_results(args.rescore)
        print("changed:", rescore(rows) or "nothing")
        DECOMPOSE = any(" || " in (r.get("queries") or "") for r in rows)
        summary = summarize(rows, k)
        write_report(rows, summary, k, args.rescore)
        for m, v in summary.items():
            print(f"| {m} | {fmt(v)} |")
        return

    if args.retrieval_only is not None:
        tests = load_tests(args.testset)
        models = ["none"] + [m for m in args.retrieval_only if m.lower() != "none"]
        caps = args.max_per_source or [config.MAX_PER_SOURCE]
        decomposer = Decomposer(Generator()) if args.decompose else None
        table = {}
        for m in models:
            for cap in caps:
                t0 = time.time()
                summary, misses = retrieval_metrics(Retriever(rerank_model=m, max_per_source=cap), tests, k,
                                                    decomposer)
                label = m if len(caps) == 1 else f"{m}, max {cap or '∞'}/paper"
                label += " + decomposition" if decomposer else ""
                table[label] = (summary, misses, (time.time() - t0) / len(tests))
        metrics = list(next(iter(table.values()))[0])
        print("\n| Retriever | " + " | ".join(metrics) + " | s/question | page misses |")
        print("|---" * (len(metrics) + 3) + "|")
        for m, (sm, miss, sec) in table.items():
            print(f"| {m} | " + " | ".join(fmt(sm[x]) for x in metrics) + f" | {sec:.2f} | {' '.join(miss)} |")
        return

    RERANK = args.rerank_model
    DECOMPOSE = args.decompose
    rows = evaluate(k, RERANK, args.testset, DECOMPOSE)
    summary = summarize(rows, k)
    write_report(rows, summary, k, suffix)
    print("\n| Metric | Value |\n|---|---|")
    for m, v in summary.items():
        print(f"| {m} | {fmt(v)} |")
    print(f"\nWrote {EVAL_DIR / f'report{suffix}.md'} and results{suffix}.csv")


if __name__ == "__main__":
    main()
