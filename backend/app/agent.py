"""Claude-powered math agent.

Architecture: the model IS the router. Claude gets these capabilities and
decides per question:

  * search_knowledge_base  — local BM25 over curated worked problems
  * SymPy tools            — deterministic computation & answer verification
                             (core + matrices/vectors/complex/systems/inequalities)
  * check_step             — numeric spot-check of each algebraic step
  * plot_function          — rendered graphs served to the frontend
  * web_search             — Anthropic's server-side web search

On top of the loop:
  * Self-correction: if the final answer wasn't machine-verified, the agent is
    sent back once to verify (and fix) it before the user sees anything.
  * Cascade (optional): solve with a cheap model first; escalate to the main
    model when the cheap answer fails verification.
  * Calibrated confidence: benchmark-measured accuracy for verified vs
    unverified answers is attached to every response.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator

import anthropic

from app.calibration import confidence_for
from app.config import settings
from app.feedback import get_few_shot_examples
from app.schemas import PracticeProblem, Solution, SolveResponse, TokenUsage
from app.tools.advanced_tools import ADVANCED_TOOLS
from app.tools.knowledge_base import KB_TOOL_SCHEMA, search_knowledge_base
from app.tools.math_tools import SYMPY_TOOLS
from app.tools.plotting import PLOT_TOOL_SCHEMA, plot_function
from app.tools.verification import CHECK_STEP_SCHEMA, check_step

logger = logging.getLogger(__name__)

# USD per 1M tokens (input, output); cache read ~0.1x input, cache write ~1.25x
_PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}

SOLVE_SYSTEM = """You are a mathematics professor who produces rigorous, \
student-friendly, step-by-step solutions.

Rules:
1. For standard textbook problems, call search_knowledge_base first and use a \
matching worked solution as your reference if relevance is high.
2. NEVER do arithmetic or algebra purely in your head: use the SymPy tools \
(solve_equation, differentiate, integrate_expression, evaluate_expression, \
simplify_expression, solve_system, solve_inequality, matrix_operation, \
vector_operation, complex_operation) to compute AND to verify your final \
answer. A solution whose final answer was not cross-checked with a SymPy tool \
is incomplete, unless the question is conceptual (proofs, definitions, open \
problems).
3. Verify each nontrivial algebraic manipulation with check_step (LHS vs RHS). \
A step that fails check_step is wrong — fix it before continuing.
4. When a graph would aid understanding (functions, areas, curves), call \
plot_function; the plot is shown to the student.
5. Use web_search only for questions about recent research, open conjectures, \
history, or applications — not for computations.
6. You only answer mathematics questions. If the question is not mathematics, \
say so briefly.
7. If a figure/diagram image is attached, read it carefully — extract all \
labels, lengths, angles and geometric relationships before solving.
8. For multiple-choice questions, begin final_answer with the correct option \
letter(s), e.g. "C" or "A, D", followed by the value.

When you have finished (after tool use), respond with ONLY a JSON object:
{
  "steps": [{"step_number": 1, "description": "...", "formula": "... or null", "explanation": "..."}],
  "final_answer": "...",
  "verification": {"verified": true/false, "method": "how it was checked", "details": "..."},
  "references": ["kb id, URL, or theorem name"],
  "diagnosis": null
}
No prose outside the JSON."""

DIAGNOSE_SYSTEM = """You are a mathematics professor reviewing a student's OWN \
worked attempt at a problem (text and/or an attached photo of their work).

Your job:
1. Reconstruct the student's steps from their work.
2. Find the FIRST step that is mathematically wrong. Verify your claim with \
the SymPy tools (check_step on their claimed equality, solve_equation, etc.) — \
never assert an error you have not machine-checked.
3. Identify the underlying misconception (sign error, wrong formula, invalid \
operation, dropped case, ...).
4. Produce the corrected solution from that point on, verifying with tools.

Respond with ONLY a JSON object:
{
  "steps": [corrected solution steps: {"step_number", "description", "formula", "explanation"}],
  "final_answer": "the correct final answer",
  "verification": {"verified": true/false, "method": "...", "details": "..."},
  "references": [],
  "diagnosis": {"error_step": <1-based step number in the STUDENT's work, or null if their work is fully correct>, "misconception": "one-sentence explanation"}
}
No prose outside the JSON."""

PRACTICE_SYSTEM = """You are a JEE mathematics examiner creating ONE new, \
original practice problem.

Rules:
1. Invent a fresh problem on the requested topic and difficulty — do not copy \
a known past-paper question. Vary the numbers and structure.
2. SOLVE your own problem completely, verifying every computation and the \
final answer with the SymPy tools (solve_equation, matrix_operation, \
check_step, ...). If your intended answer does not verify, FIX the problem or \
the solution until it does — never ship an unverified answer key.
3. The problem must be self-contained and unambiguous with a single answer.

When finished (after tool use), respond with ONLY a JSON object:
{
  "problem": "the full problem statement",
  "steps": [{"step_number": 1, "description": "...", "formula": "... or null", "explanation": "..."}],
  "final_answer": "...",
  "verification": {"verified": true/false, "method": "...", "details": "..."}
}
No prose outside the JSON."""

RETRY_PROMPT = """Your final answer was NOT machine-verified. Before answering: \
(1) independently verify or correct your final answer using the SymPy tools, \
(2) run check_step on the key algebraic steps. If a tool contradicts your \
answer, the tool is right — rework the solution. Then output the (corrected) \
JSON solution."""

_VERIFIER_TOOLS = SYMPY_TOOLS | ADVANCED_TOOLS  # count as machine verification
_VERIFIER_TOOLS = dict(_VERIFIER_TOOLS)
_VERIFIER_TOOLS["check_step"] = (check_step, CHECK_STEP_SCHEMA)

_CLIENT_TOOLS: dict[str, Any] = {
    "search_knowledge_base": search_knowledge_base,
    "plot_function": plot_function,
}
_CLIENT_TOOLS.update({name: fn for name, (fn, _s) in _VERIFIER_TOOLS.items()})

_TOOL_SCHEMAS: list[dict] = (
    [KB_TOOL_SCHEMA, PLOT_TOOL_SCHEMA]
    + [schema for _fn, schema in _VERIFIER_TOOLS.values()]
)

_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": settings.MAX_WEB_SEARCHES,
}

_PLOT_URL_RE = re.compile(r"/plots/[\w-]+\.png")

_SOLUTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "description": {"type": "string"},
                    "formula": {"type": ["string", "null"]},
                    "explanation": {"type": "string"},
                },
                "required": ["step_number", "description", "formula", "explanation"],
                "additionalProperties": False,
            },
        },
        "final_answer": {"type": "string"},
        "verification": {
            "type": "object",
            "properties": {
                "verified": {"type": "boolean"},
                "method": {"type": "string"},
                "details": {"type": ["string", "null"]},
            },
            "required": ["verified", "method", "details"],
            "additionalProperties": False,
        },
        "references": {"type": "array", "items": {"type": "string"}},
        "diagnosis": {
            "type": ["object", "null"],
            "properties": {
                "error_step": {"type": ["integer", "null"]},
                "misconception": {"type": ["string", "null"]},
            },
            "required": ["error_step", "misconception"],
            "additionalProperties": False,
        },
    },
    "required": ["steps", "final_answer", "verification", "references", "diagnosis"],
    "additionalProperties": False,
}

_PRACTICE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "steps": _SOLUTION_JSON_SCHEMA["properties"]["steps"],
        "final_answer": {"type": "string"},
        "verification": _SOLUTION_JSON_SCHEMA["properties"]["verification"],
    },
    "required": ["problem", "steps", "final_answer", "verification"],
    "additionalProperties": False,
}

_CONCEPTUAL_MARKERS = ("conceptual", "proof", "definition", "open problem",
                       "not applicable", "theoretical", "qualitative")


def needs_verification_retry(solution: Solution, sympy_ok: bool) -> bool:
    """Should the agent be sent back to verify its answer?

    Retry when the model claims verification that the server never observed,
    or when the answer is unverified for no stated conceptual reason.
    """
    v = solution.verification
    if v.verified and not sympy_ok:
        return True  # self-reported verification with no tool run — don't trust it
    if not v.verified:
        text = f"{v.method} {v.details or ''}".lower()
        return not any(marker in text for marker in _CONCEPTUAL_MARKERS)
    return False


def should_escalate(solution: Solution | None, sympy_ok: bool) -> bool:
    """Cascade: escalate to the main model when the cheap pass isn't trustworthy."""
    if solution is None:
        return True
    return not (solution.verification.verified and sympy_ok)


def estimate_cost(model: str, usage: TokenUsage) -> float:
    p_in, p_out = _PRICES.get(model, (5.0, 25.0))
    cost = (usage.input * p_in + usage.output * p_out
            + usage.cache_read * p_in * 0.1 + usage.cache_write * p_in * 1.25) / 1e6
    return round(cost, 4)


class MathAgent:
    def __init__(self) -> None:
        kwargs: dict[str, Any] = {}
        if settings.ANTHROPIC_API_KEY:
            kwargs["api_key"] = settings.ANTHROPIC_API_KEY
        # With no explicit key the SDK resolves ambient credentials
        # (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile).
        self.client = anthropic.AsyncAnthropic(**kwargs)

    def _system(self, mode: str = "solve") -> list[dict]:
        base = DIAGNOSE_SYSTEM if mode == "diagnose" else SOLVE_SYSTEM
        blocks = [{
            "type": "text",
            "text": base,
            "cache_control": {"type": "ephemeral"},
        }]
        examples = get_few_shot_examples()
        if examples:
            shots = "\n\n".join(
                f"Example of a highly-rated answer:\nQ: {e['question']}\nFinal answer: {e['final_answer']}"
                for e in examples
            )
            blocks.append({"type": "text", "text": shots})
        return blocks

    async def _agent_loop(
        self, messages: list[dict], system: list[dict], effort: str,
        model: str, ctx: dict, max_turns: int,
    ) -> AsyncIterator[dict]:
        """Core tool loop. Mutates `messages` and the ctx counters; sets
        ctx['final_text'] on success or ctx['failed'] on error/budget-exhaustion."""
        for _ in range(max_turns):
            ctx["turns"] += 1
            try:
                async with self.client.messages.stream(
                    model=model,
                    max_tokens=settings.MAX_TOKENS,
                    output_config={"effort": effort},
                    system=system,
                    tools=_TOOL_SCHEMAS + [_WEB_SEARCH_TOOL],
                    messages=messages,
                ) as stream:
                    response = await stream.get_final_message()
            except (anthropic.AuthenticationError, TypeError):
                yield {"type": "error",
                       "message": "No valid Anthropic credentials. Set ANTHROPIC_API_KEY "
                                  "in backend/.env or run `ant auth login`."}
                ctx["failed"] = True
                return
            except anthropic.APIStatusError as e:
                yield {"type": "error", "message": f"Claude API error ({e.status_code}): {e.message}"}
                ctx["failed"] = True
                return
            except anthropic.APIConnectionError:
                yield {"type": "error", "message": "Could not reach the Claude API (network error)."}
                ctx["failed"] = True
                return
            except Exception as e:  # noqa: BLE001 — never crash the SSE stream
                logger.exception("Unexpected agent error")
                yield {"type": "error", "message": f"Unexpected error: {e}"}
                ctx["failed"] = True
                return

            u = response.usage
            usage: TokenUsage = ctx["usage"]
            usage.input += u.input_tokens or 0
            usage.output += u.output_tokens or 0
            usage.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
            usage.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

            if any(b.type == "server_tool_use" for b in response.content):
                ctx["web_used"] = True
                yield {"type": "tool_call", "tool": "web_search",
                       "message": "Searching the web..."}

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            if response.stop_reason == "refusal":
                yield {"type": "error", "message": "The model declined to answer this request."}
                ctx["failed"] = True
                return

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                ctx["final_text"] = "".join(
                    b.text for b in response.content if b.type == "text")
                return

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                fn = _CLIENT_TOOLS.get(block.name)
                yield {"type": "tool_call", "tool": block.name, "input": dict(block.input)}
                if fn is None:
                    result, is_error = f"error: unknown tool {block.name}", True
                else:
                    try:
                        result = fn(**block.input)
                        is_error = result.startswith("error:")
                    except TypeError as e:
                        result, is_error = f"error: bad arguments: {e}", True
                if not is_error:
                    if block.name in _VERIFIER_TOOLS:
                        ctx["sympy_ok"] = True
                    if block.name == "check_step" and "VERIFIED" in result:
                        ctx["steps_checked"] += 1
                    if block.name == "plot_function":
                        m = _PLOT_URL_RE.search(result)
                        if m:
                            ctx["plots"].append(m.group(0))
                            yield {"type": "plot", "url": m.group(0)}
                ctx["tools_used"].append(block.name)
                yield {"type": "tool_result", "tool": block.name,
                       "preview": result[:300], "is_error": is_error}
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})

        yield {"type": "error",
               "message": f"Gave up after {ctx['turns']} tool turns."}
        ctx["failed"] = True

    async def solve_events(
        self,
        question: str,
        image_base64: str | None = None,
        image_media_type: str = "image/png",
        effort: str | None = None,
        mode: str = "solve",
        _model: str | None = None,
    ) -> AsyncIterator[dict]:
        """Run the agent, yielding progress events and a final result event."""
        start = time.monotonic()
        effort = effort or settings.EFFORT
        cascading = settings.ENABLE_CASCADE and _model is None
        model = _model or (settings.CASCADE_MODEL if cascading else settings.MODEL)

        if image_base64:
            content: list[dict] | str = [
                {"type": "image",
                 "source": {"type": "base64", "media_type": image_media_type,
                            "data": image_base64}},
                {"type": "text", "text": question},
            ]
        else:
            content = question
        messages: list[dict] = [{"role": "user", "content": content}]
        system = self._system(mode)

        ctx: dict = {"tools_used": [], "sympy_ok": False, "web_used": False,
                     "steps_checked": 0, "plots": [], "turns": 0,
                     "usage": TokenUsage(), "failed": False, "final_text": ""}

        yield {"type": "status", "message": "Thinking about the question..."}

        solution: Solution | None = None
        corrected = False
        for attempt in range(2):  # initial pass + at most one self-correction
            async for event in self._agent_loop(
                    messages, system, effort, model, ctx, settings.MAX_TOOL_TURNS):
                yield event
            if ctx["failed"]:
                return

            yield {"type": "status", "message": "Formatting the solution..."}
            try:
                solution = await self._parse_solution(question, ctx["final_text"])
            except Exception as e:  # noqa: BLE001
                yield {"type": "error", "message": f"Could not format the solution: {e}"}
                return

            if (attempt == 0 and mode == "solve" and settings.SELF_CORRECT
                    and needs_verification_retry(solution, ctx["sympy_ok"])):
                corrected = True
                yield {"type": "status",
                       "message": "Answer not machine-verified — running self-check..."}
                messages.append({"role": "assistant", "content": ctx["final_text"]})
                messages.append({"role": "user", "content": RETRY_PROMPT})
                continue
            break

        if solution is None:
            yield {"type": "error", "message": "Agent produced no solution."}
            return

        # Cascade: cheap model's answer didn't verify — escalate to the main model
        if cascading and should_escalate(solution, ctx["sympy_ok"]):
            yield {"type": "status",
                   "message": f"Escalating from {model} to {settings.MODEL}..."}
            async for event in self.solve_events(
                    question, image_base64, image_media_type, effort, mode,
                    _model=settings.MODEL):
                yield event
            return

        kb_used = "search_knowledge_base" in ctx["tools_used"]
        if ctx["web_used"]:
            ctx["tools_used"].append("web_search")
        source = ("hybrid" if kb_used and ctx["web_used"]
                  else "knowledge_base" if kb_used
                  else "web_search" if ctx["web_used"]
                  else "reasoning")

        result = SolveResponse(
            question=question,
            solution=solution,
            source=source,
            tools_used=sorted(set(ctx["tools_used"])),
            sympy_verified=ctx["sympy_ok"],
            steps_checked=ctx["steps_checked"],
            corrected=corrected,
            confidence=confidence_for(ctx["sympy_ok"]),
            plots=ctx["plots"],
            model=model,
            usage=ctx["usage"],
            est_cost=estimate_cost(model, ctx["usage"]),
            turns=ctx["turns"],
            processing_time=round(time.monotonic() - start, 2),
        )
        yield {"type": "solution", "data": json.loads(result.model_dump_json())}

    async def practice_events(self, topic: str, difficulty: str) -> AsyncIterator[dict]:
        """Generate one practice problem with a machine-verified answer key."""
        prompt = (f"Create one {difficulty} level problem on the topic: {topic}. "
                  f"Solve and verify it as instructed.")
        messages: list[dict] = [{"role": "user", "content": prompt}]
        system = [{"type": "text", "text": PRACTICE_SYSTEM,
                   "cache_control": {"type": "ephemeral"}}]
        ctx: dict = {"tools_used": [], "sympy_ok": False, "web_used": False,
                     "steps_checked": 0, "plots": [], "turns": 0,
                     "usage": TokenUsage(), "failed": False, "final_text": ""}

        yield {"type": "status", "message": f"Creating a {difficulty} problem on {topic}..."}
        async for event in self._agent_loop(
                messages, system, settings.EFFORT, settings.MODEL, ctx,
                settings.MAX_TOOL_TURNS):
            yield event
        if ctx["failed"]:
            return

        try:
            problem = await self._parse_json(
                PracticeProblem, _PRACTICE_JSON_SCHEMA,
                f"practice problem on {topic}", ctx["final_text"])
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "message": f"Could not format the problem: {e}"}
            return
        problem.topic = topic
        problem.difficulty = difficulty
        data = json.loads(problem.model_dump_json())
        data["sympy_verified"] = ctx["sympy_ok"]
        data["est_cost"] = estimate_cost(settings.MODEL, ctx["usage"])
        yield {"type": "practice", "data": data}

    async def _parse_json(self, model_cls, json_schema: dict, label: str,
                          final_text: str):
        """Parse model output into a schema; structured-output fallback pass."""
        raw = final_text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        candidate = match.group(1) if match else raw
        try:
            return model_cls.model_validate_json(candidate)
        except Exception:  # noqa: BLE001
            logger.info("Direct JSON parse failed for %s; running formatting pass", label)

        response = await self.client.messages.create(
            model=settings.MODEL,
            max_tokens=settings.MAX_TOKENS,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": json_schema},
            },
            messages=[{
                "role": "user",
                "content": f"Convert this {label} to the required JSON format.\n\n"
                           f"{final_text}",
            }],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return model_cls.model_validate_json(text)

    async def _parse_solution(self, question: str, final_text: str) -> Solution:
        """Parse the model's JSON; fall back to a structured-output formatting pass."""
        raw = final_text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        candidate = match.group(1) if match else raw
        try:
            return Solution.model_validate_json(candidate)
        except Exception:  # noqa: BLE001 — any parse failure goes to the fallback
            logger.info("Direct JSON parse failed; running formatting pass")

        response = await self.client.messages.create(
            model=settings.MODEL,
            max_tokens=settings.MAX_TOKENS,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _SOLUTION_JSON_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": f"Convert this solution to the required JSON format.\n\n"
                           f"Question: {question}\n\nSolution:\n{final_text}",
            }],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return Solution.model_validate_json(text)


math_agent = MathAgent()
