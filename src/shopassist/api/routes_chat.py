"""Story 0.3. POST /chat returning an SSE StreamingResponse. Translates ChatEvent objects to SSE frames. No model logic here."""

import logging
from asyncio import CancelledError
from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from shopassist.api.deps import GatewayDependency, LedgerDependency, SessionDependency
from shopassist.api.schemas import (
    ChatRequest,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    format_sse,
)
from shopassist.llm.context import CallContext
from shopassist.llm.errors import AttemptFailure
from shopassist.llm.events import Completion, TextDelta
from shopassist.llm.usage import Purpose

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", tags=["chat"])
async def chat(
    req: ChatRequest,
    gateway: GatewayDependency,
    ledger: LedgerDependency,
    session: SessionDependency,
) -> StreamingResponse:
    ctx = CallContext(purpose=Purpose.CHAT, session_id=session.session_id)
    user_msg = session.memory.append("user", req.message)
    messages = session.memory.get_messages()

    async def event_stream() -> AsyncGenerator[str]:
        try:
            async for ev in gateway.stream(ctx, req.tier, messages):
                if isinstance(ev, TextDelta):
                    yield format_sse("delta", DeltaEvent(text=ev.text))
                elif isinstance(ev, Completion):
                    session.memory.append("assistant", ev.text)
                    yield format_sse(
                        "done",
                        DoneEvent(
                            reply_length=len(ev.text),
                            prompt_tokens=ev.record.prompt_tokens,
                            completion_tokens=ev.record.completion_tokens,
                            cost_scaled=ev.record.cost_scaled,
                            served_model=ev.record.served_model,
                        ),
                    )
        except AttemptFailure as exc:
            yield format_sse(
                "error",
                ErrorEvent(kind=exc.kind.value, chunks_received=exc.chunks_received),
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session.session_id, "Cache-Control": "no-cache"},
    )
