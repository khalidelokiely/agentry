"""Integration tests for FastAPI SSE /chat service, session isolation, rollback, and concurrency locks."""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any, AsyncGenerator
from uuid import uuid7

import pytest
from httpx import ASGITransport, AsyncClient

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


class FakeStreamingGateway: ...
