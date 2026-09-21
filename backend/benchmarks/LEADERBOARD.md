# JEE Math Leaderboard

Reproducible accuracy on the JEEBench math subset (236 JEE Advanced problems,
2016–2023). Every row is reproduced with a single command — no hand-picked
numbers.

| Model | Effort | Problems | Accuracy | Verified acc. | Unverified acc. | Cost | Command |
|---|---|---|---|---|---|---|---|
| claude-opus-5 | xhigh | _pending_ | — | — | — | — | `python -m benchmarks.run_jee_benchmark --effort xhigh --yes` |
| claude-sonnet-5 | xhigh | _pending_ | — | — | — | — | `python -m benchmarks.run_jee_benchmark --model claude-sonnet-5 --effort xhigh --fresh --yes` |
| claude-haiku-4-5 | high | _pending_ | — | — | — | — | `python -m benchmarks.run_jee_benchmark --model claude-haiku-4-5 --effort high --fresh --yes` |

Fill a row by running its command (requires `ANTHROPIC_API_KEY`), then copying
the numbers from `results/REPORT.md`. Use `--limit 50` for a cheaper sample —
note the reduced N in the Problems column.

Methodology: single attempt per problem, self-correction enabled, grading per
`benchmarks/grader.py` (exact letter/set match, integer exact, numeric ±0.01;
partial credit tracked separately for multi-select).
