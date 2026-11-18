"""Integration tests for FastAPI SSE /chat service, session isolation, rollback, and concurrency locks."""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any, AsyncGenerator
from uuid import uuid7

import pytest
from httpx import ASGITransport, AsyncClient
from litellm.llms.custom_httpx.http_handler import headers

from shopassist.api.app import create_app
from shopassist.llm.context import CallContext
from shopassist.llm.events import Completion, LLMEvent, TextDelta
from shopassist.llm.gateway import LLMGateway
from shopassist.llm.models import Tier
from shopassist.llm.usage import CallRecord, Outcome, Purpose, UsageLedger


def _make_record(
    ctx: CallContext, tier: Tier, outcome: Outcome, chunks: int
) -> CallRecord:
    return CallRecord(
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


@asynccontextmanager
async def make_client(delay_s: float = 0.01):
    ledger = UsageLedger()
    fake_gateway = FakeStreamingGateway(ledger, delay_s=delay_s)
    app = create_app(fake_gateway, ledger)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app, ledger


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
            rec = _make_record(ctx, tier, outcome, chunks)
            self._ledger.record(rec)
        yield Completion(text=reply, record=rec)


@pytest.mark.asyncio
async def test_multi_session_memory_isolation() -> None:
    async with make_client(delay_s=0.01) as (client, app, ledger):
        r1 = await client.post(
            "/chat", headers={"X-Session-Id": "s1"}, json={"message": "alpha"}
        )
        r2 = await client.post(
            "/chat", headers={"X-Session-Id": "s2"}, json={"message": "beta"}
        )
        r3 = await client.post(
            "/chat", headers={"X-Session-Id": "s1"}, json={"message": "gamma"}
        )
        assert r1.status_code == 200 and "turns=1" in r1.text
        assert r2.status_code == 200 and "turns=1" in r2.text
        assert r3.status_code == 200 and "turns=2" in r3.text


@pytest.mark.asyncio
async def test_disconnect_rollback_and_concurrency_lock() -> None:
    async with make_client(delay_s=0.15) as (client, app, ledger):
        task = asyncio.create_task(
            client.post(
                "/chat",
                headers={"X-Session-Id": "s1"},
                json={"message": "hold one two"},
            )
        )
        await asyncio.sleep(0.03)
        conflict = await client.post(
            "/chat", headers={"X-Session-Id": "s1"}, json={"message": "overlap"}
        )
        assert conflict.status_code == 409
        parallel = await client.post(
            "/chat", headers={"X-Session-Id": "s2"}, json={"message": "parallel"}
        )
        assert parallel.status_code == 200 and "turns=1" in parallel.text

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        s1_records = ledger.by_session("s1")
        assert len(s1_records) == 1
        assert s1_records[-1].outcome == Outcome.CANCELLED
        assert s1_records[-1].cost_scaled > 0

        s1_session = app.state.sessions.get_or_create("s1")
        assert len(s1_session.memory.view()) == 0
        assert not s1_session.lock.locked()


@pytest.mark.asyncio
async def test_empty_payload_rejected_422() -> None:
    async with make_client() as (client, app, ledger):
        resp = await client.post(
            "/chat", headers={"X-Session-Id": "s1"}, json={"message": ""}
        )
        assert resp.status_code == 422
        assert len(ledger.records) == 0
