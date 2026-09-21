# Math Agent v2 — Agentic Math Tutor

A working rebuild of [captflag/math-agent-system](https://github.com/captflag/math-agent-system):
a math professor agent that produces step-by-step, **machine-verified** solutions.

## What changed from v1

| v1 (original repo) | v2 (this rebuild) |
|---|---|
| LangGraph router deciding "KB vs web" with an extra LLM call | **The model is the router** — Claude picks tools per question, no state machine |
| Qdrant + sentence-transformers (never wired up) | Local BM25 index over a bundled worked-problem KB — zero services, swappable for a vector DB later |
| MCP server + Tavily for web search (didn't exist) | **Anthropic server-side `web_search` tool** — no key, no server, runs on Anthropic infra |
| LLM response thrown away (`"steps": []` hardcoded) | Structured `Solution` schema; JSON parse with a guaranteed structured-output fallback pass |
| Regex "hallucination detection" | **SymPy verification**: the agent must cross-check answers with symbolic tools; the server independently records whether a SymPy tool actually ran (`sympy_verified` badge) |
| DSPy dependency (unused) | Feedback → SQLite → top-rated answers injected as few-shot exemplars |
| Endpoints echoed the request back | Real SSE streaming of agent progress + solve/feedback/health APIs |
| Presidio (heavy, unwired) | Lightweight input guardrails: topic check, length, PII scrubbing |

## What makes it different

- **Self-correcting solves**: an answer that wasn't machine-verified is sent
  back to the agent once with the discrepancy before the user ever sees it.
- **Step-level verification**: `check_step` numerically spot-checks each
  algebraic manipulation at random points — every line gets a ✓/✗, not just
  the conclusion.
- **Calibrated confidence**: benchmark runs measure how often verified vs
  unverified answers were actually correct; those measured rates ship with
  every future answer.
- **Mistake diagnosis**: upload your own worked attempt (text or photo) and
  the agent finds the first wrong step — verified with SymPy, never asserted.
- **Socratic tutor**: multi-turn guided sessions that check your algebra with
  tools but never hand over the answer.
- **Practice generator**: fresh JEE-style problems whose answer keys are
  machine-verified before they're served.
- **Cost cascade (opt-in)**: solve with Haiku first, escalate to Opus only
  when the cheap answer fails verification. Every response carries its token
  usage and estimated cost.
- **Graphs**: the agent plots functions (matplotlib, server-side) when a
  picture helps; plots render inline in the frontend.

## Architecture

```
question (+ optional figure image, + effort level)
        → input guardrails (topic / length / PII scrub; image bypasses topic check)
        → Claude Opus 5 agentic loop (SSE progress events; 'xhigh' effort for JEE Advanced)
             ├─ search_knowledge_base   (local BM25, 64 JEE-syllabus worked problems)
             ├─ SymPy core tools        (solve / differentiate / integrate / evaluate / simplify)
             ├─ SymPy advanced tools    (systems, inequalities, matrices, vectors, complex numbers)
             └─ web_search              (Anthropic server-side tool)
        → structured Solution JSON (fallback: structured-output formatting pass)
        → response with sympy_verified + source + tools_used metadata
feedback → SQLite → few-shot exemplars in the system prompt
```

## JEE benchmark

`benchmarks/jee_math.json` bundles the math subset of
[JEEBench](https://huggingface.co/datasets/daman1209arora/jeebench) — 236 real
JEE Advanced problems (2016–2023) with gold answers across four types
(MCQ, multi-select MCQ, Integer, Numeric).

```bash
cd backend
python -m benchmarks.run_jee_benchmark --limit 50 --effort xhigh --yes
```

- Grades per type (letter match, set match with partial credit, exact integer,
  numeric ±0.01) and reports accuracy overall, by type, and split by the
  `sympy_verified` flag — measuring whether machine verification actually
  predicts correctness.
- Results stream to `benchmarks/results/results.jsonl`; interrupted runs resume
  with `--resume`. Report lands in `benchmarks/results/REPORT.md`.
- Needs `ANTHROPIC_API_KEY`; prints an estimated cost and asks before spending
  (skip with `--yes`). A 50-problem run at Opus 5 pricing is roughly $9.

Security note: SymPy's `parse_expr` ultimately `eval()`s, so every expression is
validated first (character whitelist, no dunders, identifier allow-list). There is
a regression test proving `__import__('os')...` is rejected.

## Run it

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env       # put your ANTHROPIC_API_KEY here (or use `ant auth login`)
python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000 — the frontend (React + Tailwind + shadcn/ui, bundled to
a single file in `frontend/index.html`) is served by the backend. API docs at `/api/docs`.

To change the UI, edit `frontend-src/` and rebuild:

```bash
cd frontend-src
pnpm install
pnpm exec vite build   # then inline dist into ../frontend/index.html (see frontend-src/README)
```

Or with Docker: `docker build -t math-agent-v2 . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... math-agent-v2`

## API

- `POST /api/v1/solve` — SSE stream: `status`, `tool_call`, `tool_result`, `plot`, `solution`, `error`, `done`. Body accepts `mode: "diagnose"`, `effort`, `image_base64`.
- `POST /api/v1/solve/sync` — same pipeline, final JSON only
- `POST /api/v1/tutor` — Socratic tutoring turn; pass back `session_id` for multi-turn
- `POST /api/v1/practice` — generate a problem with a verified answer key
- `GET /api/v1/calibration` — benchmark-measured verified/unverified accuracy
- `POST /api/v1/feedback`, `GET /api/v1/feedback/stats`, `GET /api/v1/health`

Solve/tutor/practice are rate-limited per IP (default 30/min).

## Tests

```bash
cd backend && python -m pytest tests/ -q
```

81 offline tests (no API key needed): SymPy core + advanced tools with
injection hardening, step-level verification (`check_step` catches planted
sign errors), self-correction and cascade decision logic, calibration math,
plotting, rate limiting, tutor sessions, BM25 retrieval, guardrails, benchmark
grader, schemas, feedback store, API validation.

After a benchmark run, `benchmarks/LEADERBOARD.md` holds the reproducible
accuracy table and `data/calibration.json` powers the confidence badges.
