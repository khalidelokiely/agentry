"""Story 0.3. POST /chat returning an SSE StreamingResponse. Translates ChatEvent objects to SSE frames. No model logic here."""
from asyncio import CancelledError
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from litellm import FalAIError

from shopassist.api.deps import GatewayDependency, SessionDependency, LedgerDependency
from shopassist.api.schemas import ChatRequest, DeltaEvent, DoneEvent, ErrorEvent, format_sse
from shopassist.llm.context import CallContext
from shopassist.llm.errors import AttemptFailure
from shopassist.llm.events import Completion, TextDelta
from shopassist.llm.usage import Purpose
from contextlib import aclosing

from shopassist.memory import messages

log = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", tags=["chat"])
async def chat(req: ChatRequest, gateway: GatewayDependency, ledger: LedgerDependency, session: SessionDependency) -> StreamingResponse:
    if session.lock.locked():
        raise HTTPException(status_code=409, detail=f"Session {session.session_id} already has an active stream.")
    await session.lock.acquire()

    ctx = CallContext(purpose=Purpose.CHAT, session_id=session.session_id)
    user_msg = session.memory.append(role="user", content=req.message)
    messages = session.memory.get_messages()

    async def event_stream() -> AsyncGenerator[str, None]:
        completed = False
        try:
            async with aclosing(gateway.stream(ctx, req.tier, messages)) as stream:
                async for ev in stream:
                    if isinstance(ev, TextDelta):
                        yield format_sse("delta", DeltaEvent(text=ev.text))
                    elif isinstance(ev, Completion):
                        session.memory.append(role="assistant", content=ev.text)
                        completed = True
                        yield format_sse(
                            "done",
                            DoneEvent(
                                reply_length=len(ev.text),
                                prompt_tokens=ev.record.prompt_tokens,
                                completion_tokens=ev.record.completion_tokens,
                                cost_scaled=ev.record.cost_scaled,
                                served_model=ev.record.served_model,
                            )
                        )
        except AttemptFailure as exc:
            yield format_sse(
                "error",
                ErrorEvent(
                    kind=exc.kind.value,
                    chunks_received=exc.chunks_received,
                )
            )
        except (CancelledError, GeneratorExit):
            log.warning("client disconnected mid-stream session=%s", session.session_id)
        finally:
            if not completed:
                purged = session.memory.rollback(user_msg.id)
                rows = ledger.by_session(session.session_id)
                outcome = rows[-1].outcome.value if rows else "none"
                log.warning("rollback session=%s purged=%s outcome=%s", session.session_id, purged, outcome)

            session.lock.release()
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session.session_id, "Cache-Control": "no-cache"},

    )