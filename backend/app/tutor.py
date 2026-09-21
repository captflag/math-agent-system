"""Socratic tutor: multi-turn guided sessions that never just hand over the answer."""
from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from app.config import settings

TUTOR_SYSTEM = """You are a Socratic mathematics tutor for JEE-level students.

Rules:
1. NEVER give the full solution or final answer outright. Guide with one hint
   or one leading question at a time.
2. When the student attempts a step, check their algebra with the SymPy tools
   (check_step, solve_equation, evaluate_expression, ...) before judging it.
   Tell them precisely what is right or wrong in their step.
3. If they are stuck after two hints on the same step, reveal just that step
   and ask them to continue.
4. Only after the student reaches the answer themselves (or explicitly gives
   up), confirm the verified final answer.
5. Keep replies short — 2 to 5 sentences plus at most one formula.
6. Stay on mathematics. Respond in plain text (light Markdown), not JSON."""


class SessionStore:
    """In-memory tutor sessions with TTL expiry."""

    def __init__(self, ttl: int | None = None):
        self.ttl = ttl or settings.TUTOR_SESSION_TTL
        self._sessions: dict[str, dict] = {}

    def purge(self, now: float | None = None) -> None:
        now = now or time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s["touched"] > self.ttl]
        for sid in expired:
            del self._sessions[sid]

    def get_or_create(self, session_id: str | None) -> tuple[str, list[dict]]:
        self.purge()
        if session_id and session_id in self._sessions:
            s = self._sessions[session_id]
            s["touched"] = time.time()
            return session_id, s["messages"]
        sid = session_id or uuid.uuid4().hex
        self._sessions[sid] = {"messages": [], "touched": time.time()}
        return sid, self._sessions[sid]["messages"]

    def __len__(self) -> int:
        return len(self._sessions)


session_store = SessionStore()


async def tutor_events(
    message: str,
    session_id: str | None = None,
    image_base64: str | None = None,
    image_media_type: str = "image/png",
) -> AsyncIterator[dict]:
    """One tutor turn: append the student message, run the agent loop with the
    Socratic system prompt, reply in plain text."""
    from app.agent import math_agent  # late import to avoid a cycle

    sid, messages = session_store.get_or_create(session_id)
    yield {"type": "session", "session_id": sid}

    if image_base64:
        content: list[dict] | str = [
            {"type": "image",
             "source": {"type": "base64", "media_type": image_media_type,
                        "data": image_base64}},
            {"type": "text", "text": message},
        ]
    else:
        content = message
    messages.append({"role": "user", "content": content})

    system = [{"type": "text", "text": TUTOR_SYSTEM,
               "cache_control": {"type": "ephemeral"}}]
    from app.schemas import TokenUsage
    ctx: dict = {"tools_used": [], "sympy_ok": False, "web_used": False,
                 "steps_checked": 0, "plots": [], "turns": 0,
                 "usage": TokenUsage(), "failed": False, "final_text": ""}

    async for event in math_agent._agent_loop(
            messages, system, "medium", settings.MODEL, ctx,
            settings.MAX_TOOL_TURNS):
        yield event
    if ctx["failed"]:
        messages.pop()  # keep the session consistent after a failed turn
        return

    reply = ctx["final_text"].strip()
    messages.append({"role": "assistant", "content": reply})
    yield {"type": "tutor_reply", "text": reply, "plots": ctx["plots"]}
