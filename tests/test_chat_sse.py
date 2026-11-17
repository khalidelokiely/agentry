"""Integration tests for FastAPI SSE /chat service, session isolation, rollback, and concurrency locks."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

import pytest
from httpx import ASGITransport, AsyncClient

from shopassist.api.app import create_app
from shopassist.llm.context import CallContext
from shopassist.llm.events import Completion, LLMEvent, TextDelta
from shopassist.llm.models import Tier
from shopassist.llm.usage import CallRecord, Outcome, Purpose, UsageLedger


class FakeStreamingGateway:
    def __init__(self, ledger: UsageLedger, delay_s: float = 0.01) -> None:
        self._ledger = ledger
        self._delay_s = delay_s

    async def stream(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> AsyncGenerator[LLMEvent]:
        outcome = Outcome.CANCELLED
        chunks = 0
        try:
            user_texts = [m["content"] for m in messages if m["role"] == "user"]
            reply = f"turns={len(user_texts)} last={user_texts[-1]}"
            for word in reply.split(" "):
                await asyncio.sleep(self._delay_s)
                chunks += 1
                yield TextDelta(text=word + " ")
            outcome = Outcome.OK
        finally:
            rec = CallRecord(
                call_id=uuid7().hex,
                turn_id=ctx.turn_id or uuid7().hex,
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                purpose=ctx.purpose,
                cluster="test",
                requested_model=tier.value,
                served_model="gemini-3.5-flash-lite",
                prompt_tokens=10,
                completion_tokens=chunks,
                latency_s=0.05,
                ttfc_s=0.01,
                cost_scaled=chunks * 1000,
                estimated=False,
                outcome=outcome,
                started_at=datetime.now(UTC),
                attempt_index=0,
                chunks_received=chunks,
                failure_kind=None,
            )
            self._ledger.record(rec)
        yield Completion(text=reply, record=rec)

    async def complete(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> Completion:
        raise NotImplementedError


@asynccontextmanager
async def make_client(delay_s: float = 0.01):
    ledger = UsageLedger()
    app = create_app(FakeStreamingGateway(ledger, delay_s=delay_s), ledger)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app, ledger


@pytest.mark.asyncio
async def test_multi_session_memory_isolation() -> None: ...


@pytest.mark.asyncio
async def test_disconnect_rollback_and_concurrency_lock() -> None: ...
