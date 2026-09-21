"""JEE Math benchmark runner.

Runs JEEBench math problems (JEE Advanced 2016-2023, 236 problems) through the
agent and grades against gold answers.

Usage (from backend/):
    python -m benchmarks.run_jee_benchmark --limit 50 --effort xhigh --yes
    python -m benchmarks.run_jee_benchmark --resume          # continue a run
    python -m benchmarks.run_jee_benchmark --report-only     # regenerate report

Results stream to benchmarks/results/results.jsonl (resumable); the summary
report is written to benchmarks/results/REPORT.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import MathAgent  # noqa: E402
from app.calibration import save_from_rows  # noqa: E402
from app.config import settings  # noqa: E402
from benchmarks.grader import TYPE_INSTRUCTIONS, grade  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
DATA = BENCH_DIR / "jee_math.json"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "results.jsonl"
REPORT_FILE = RESULTS_DIR / "REPORT.md"

# Rough per-problem cost guess at Opus 5 pricing (~15K in / ~4K out with tools)
EST_COST_PER_PROBLEM = 0.18


def load_problems(limit: int | None, offset: int, types: list[str] | None) -> list[dict]:
    problems = json.loads(DATA.read_text(encoding="utf-8"))
    if types:
        problems = [p for p in problems if p["type"] in types]
    problems = problems[offset:]
    return problems[:limit] if limit else problems


def load_done() -> dict[int, dict]:
    done = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["index"]] = row
    return done


async def solve_one(agent: MathAgent, problem: dict, effort: str,
                    sem: asyncio.Semaphore) -> dict:
    prompt = (f"{problem['question']}\n\n[{TYPE_INSTRUCTIONS[problem['type']]}]")
    async with sem:
        start = time.monotonic()
        solution_event = error = None
        try:
            async for event in agent.solve_events(prompt, effort=effort):
                if event["type"] == "solution":
                    solution_event = event["data"]
                elif event["type"] == "error":
                    error = event["message"]
        except Exception as e:  # noqa: BLE001
            error = str(e)
        elapsed = round(time.monotonic() - start, 1)

    row = {
        "index": problem["index"],
        "description": problem["description"],
        "type": problem["type"],
        "gold": problem["gold"],
        "elapsed": elapsed,
    }
    if solution_event is None:
        row.update({"error": error or "no solution", "correct": False,
                    "partial": False, "predicted": "", "sympy_verified": False})
        return row

    final_answer = solution_event["solution"]["final_answer"]
    graded = grade(problem["type"], problem["gold"], final_answer)
    row.update(graded)
    row["final_answer"] = final_answer[:400]
    row["sympy_verified"] = solution_event["sympy_verified"]
    row["source"] = solution_event["source"]
    row["tools_used"] = solution_event["tools_used"]
    row["steps_checked"] = solution_event.get("steps_checked", 0)
    row["corrected"] = solution_event.get("corrected", False)
    row["est_cost"] = solution_event.get("est_cost", 0.0)
    row["model"] = solution_event.get("model", "")
    return row


def build_report(rows: list[dict]) -> str:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    partial = sum(r.get("partial", False) for r in rows)
    errors = sum(1 for r in rows if r.get("error"))

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    verified = [r for r in rows if r.get("sympy_verified")]
    unverified = [r for r in rows if not r.get("sympy_verified")]

    def acc(rs: list[dict]) -> str:
        return f"{sum(x['correct'] for x in rs)}/{len(rs)} ({100 * sum(x['correct'] for x in rs) / len(rs):.1f}%)" if rs else "n/a"

    lines = [
        "# JEE Math Benchmark Report",
        "",
        f"Dataset: JEEBench math subset (JEE Advanced papers) — {n} problems run",
        "",
        f"## Overall accuracy: {acc(rows)}",
        f"- Partial credit (subset of correct options): {partial}",
        f"- Agent errors / no answer: {errors}",
        "",
        "## By problem type",
        "",
        "| Type | Accuracy |",
        "|---|---|",
    ]
    for t in sorted(by_type):
        lines.append(f"| {t} | {acc(by_type[t])} |")
    total_cost = sum(r.get("est_cost", 0.0) for r in rows)
    corrected_rows = [r for r in rows if r.get("corrected")]
    lines += [
        "",
        f"Measured API cost: ~${total_cost:.2f} · "
        f"self-correction triggered on {len(corrected_rows)} problems "
        f"({acc(corrected_rows)} of those ended correct)",
        "",
        "## SymPy verification signal",
        "",
        f"- SymPy-verified answers:     {acc(verified)}",
        f"- Not machine-verified:       {acc(unverified)}",
        "",
        "If verified accuracy is materially higher, the `sympy_verified` badge is",
        "doing its job as a trust signal.",
        "",
        "## Wrong answers",
        "",
        "| Index | Type | Gold | Predicted | Verified |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if not r["correct"]:
            lines.append(
                f"| {r['index']} | {r['type']} | {r['gold']} | "
                f"{r.get('predicted', '')} | {r.get('sympy_verified', False)} |")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="max problems to run")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--types", nargs="*", default=None,
                    choices=["MCQ", "MCQ(multiple)", "Integer", "Numeric"])
    ap.add_argument("--effort", default="xhigh",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--model", default=None,
                    help="override the model (e.g. claude-haiku-4-5 for a cheap dry run)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="skip problems already in results.jsonl")
    ap.add_argument("--fresh", action="store_true", help="delete previous results first")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.report_only:
        rows = list(load_done().values())
        if not rows:
            print("No results to report on."); return
        REPORT_FILE.write_text(build_report(rows), encoding="utf-8")
        print(f"Report written to {REPORT_FILE}"); return

    if args.fresh and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    problems = load_problems(args.limit, args.offset, args.types)
    done = load_done() if (args.resume or not args.fresh) else {}
    todo = [p for p in problems if p["index"] not in done]

    est = len(todo) * EST_COST_PER_PROBLEM
    print(f"{len(todo)} problems to run (effort={args.effort}, "
          f"concurrency={args.concurrency}); estimated cost ~${est:.2f}")
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            return

    if args.model:
        settings.MODEL = args.model
    settings.ENABLE_CASCADE = False   # benchmark one model at a time
    agent = MathAgent()
    sem = asyncio.Semaphore(args.concurrency)
    completed = 0

    async def run(p: dict) -> None:
        nonlocal completed
        row = await solve_one(agent, p, args.effort, sem)
        with RESULTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        completed += 1
        mark = "✓" if row["correct"] else "✗"
        print(f"[{completed}/{len(todo)}] {mark} #{row['index']} ({row['type']}) "
              f"gold={row['gold']} pred={row.get('predicted', '?')} {row['elapsed']}s")

    await asyncio.gather(*(run(p) for p in todo))

    rows = list(load_done().values())
    REPORT_FILE.write_text(build_report(rows), encoding="utf-8")
    cal = save_from_rows(rows)
    print(f"\nDone. {sum(r['correct'] for r in rows)}/{len(rows)} correct.")
    print(f"Report: {REPORT_FILE}")
    print(f"Calibration saved: verified={cal['verified_accuracy']}, "
          f"unverified={cal['unverified_accuracy']} → future answers now carry "
          f"calibrated confidence.")


if __name__ == "__main__":
    asyncio.run(main())
