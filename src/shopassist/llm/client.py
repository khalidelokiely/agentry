"""Story 0.1. Class: LLMClient. The ONLY caller of router.acompletion.
complete(purpose, ...) for summarizer, stream(purpose, ...) for chat. Writes a CallRecord for every call."""

import asyncio
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, AsyncGenerator
from uuid import uuid7

import litellm
from litellm import (
    APIError,
    AuthenticationError,
    BadRequestError,
    Router,
    token_counter,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.types.utils import ModelResponse

from shopassist.llm.context import CallContext
from shopassist.llm.cost import CostConfig
from shopassist.llm.errors import (
    AttemptFailure,
    CompletionTimeout,
    ConnectTimeout,
    FirstChunkTimeout,
    StreamStalled,
)
from shopassist.llm.events import Completion, LLMEvent, TextDelta
from shopassist.llm.models import Tier, Timeouts
from shopassist.llm.pricing import PricingTable, UnknownModelError
from shopassist.llm.stream_guard import StreamGuard
from shopassist.llm.usage import CallRecord, FailureKind, Outcome, UsageLedger

log = logging.getLogger(__name__)

_OVERLOADED = (litellm.RateLimitError, litellm.ServiceUnavailableError)
_REJECTED = _OVERLOADED + (litellm.InternalServerError, litellm.AuthenticationError)


@dataclass
class _Attempt:
    """Mutable scratchpad for ONE attempt. Turned into an immutable CallRecord at the end."""

    started_at: datetime
    t0: float
    parts: list[str] = field(default_factory=list)
    chunks: int = 0
    ttfc_s: float | None = None
    served: str | None = None
    usage: Any = None
    tool_call_seen: bool = False
    # Default is CANCELLED on purpose: if we exit without reaching a line that sets the
    # outcome, it was a cancellation (CancelledError / GeneratorExit), and `finally` still records it.
    outcome: Outcome = Outcome.CANCELLED
    failure_kind: FailureKind | None = None
    reached_provider: bool = False
    cause: Exception | None = None


def _classify(a: _Attempt, exc: Exception) -> None:
    a.cause = exc
    if isinstance(exc, FirstChunkTimeout):
        a.outcome, a.failure_kind = (
            Outcome.FIRST_CHUNK_TIMEOUT,
            FailureKind.FIRST_CHUNK_TIMEOUT,
        )
    elif isinstance(exc, CompletionTimeout):
        a.outcome, a.failure_kind = Outcome.TIMEOUT, FailureKind.TIMEOUT
    elif isinstance(exc, StreamStalled):
        a.outcome, a.failure_kind = Outcome.STALLED, FailureKind.STALLED
    elif isinstance(exc, _OVERLOADED):
        a.outcome, a.failure_kind = Outcome.ERROR, FailureKind.OVERLOADED
    elif isinstance(exc, ConnectTimeout):
        a.outcome, a.failure_kind = Outcome.CONNECT_TIMEOUT, FailureKind.CONNECT_TIMEOUT
    elif isinstance(exc, AuthenticationError) or (
        isinstance(exc, (BadRequestError, APIError, VertexAIError))
        and (msg := str(exc).lower())
        and any(
            ind in msg
            for ind in (
                "api key not valid",
                "api_key_invalid",
                "invalid api key",
                "incorrect api key",
                "invalid x-api-key",
                "authentication_error",
            )
        )
    ):
        a.outcome, a.failure_kind = (
            Outcome.AUTHENTICATION_ERROR,
            FailureKind.AUTHENTICATION_ERROR,
        )
    else:
        log.warning("unclassified failure: %s: %s", type(exc).__qualname__, exc)
        a.outcome, a.failure_kind = Outcome.ERROR, FailureKind.ERROR


class LLMClient:
    def __init__(
        self,
        cluster: str,
        router: Router,
        ledger: UsageLedger,
        pricing: PricingTable,
        timeouts: dict[Tier, Timeouts],
        guard: StreamGuard | None = None,
    ):
        self._cluster = cluster
        self._router = router
        self._ledger = ledger
        self._pricing = pricing
        self._timeouts = timeouts
        self._guard = StreamGuard() if guard is None else guard

    async def stream(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> AsyncGenerator[LLMEvent]:

        ctx = self._with_turn(ctx)
        a = _Attempt(started_at=datetime.now(UTC), t0=time.perf_counter())
        limits = self._timeouts[tier]

        try:
            try:
                async with asyncio.timeout(limits.stream_open_s) as cm:
                    source = await self._router.acompletion(
                        model=tier.value,
                        messages=messages,
                        stream=True,
                        stream_options={"include_usage": True},
                        **opts,
                    )
                    a.reached_provider = True
            except litellm.Timeout as exc:
                raise FirstChunkTimeout(
                    "router timed out waiting for first chunk"
                ) from exc
            except TimeoutError as exc:
                if cm.expired():
                    raise ConnectTimeout(
                        f"no response from router in {limits.stream_open_s:.1f}s"
                    ) from exc
                raise

            async with aclosing(self._guard.wrap(source, limits.stall_s)) as chunks:
                async for chunk in chunks:
                    a.chunks += 1
                    if a.ttfc_s is None:
                        a.ttfc_s = time.perf_counter() - a.t0
                    a.served = getattr(chunk, "model", None) or a.served
                    if getattr(chunk, "usage", None):
                        a.usage = chunk.usage
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if getattr(delta, "tool_calls", None):
                            a.tool_call_seen = True
                        if delta.content:
                            a.parts.append(delta.content)
                            yield TextDelta(delta.content)

            a.outcome = Outcome.OK

        except Exception as exc:
            _classify(a, exc)
        finally:
            record = self._close(a, ctx, tier, messages)

        if a.cause is not None:
            raise self._failure(a, record) from a.cause

        yield Completion(text="".join(a.parts), record=record)

    async def complete(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> Completion:
        ctx = self._with_turn(ctx)
        a = _Attempt(started_at=datetime.now(UTC), t0=time.perf_counter())

        limits = self._timeouts[tier]

        try:
            try:
                async with asyncio.timeout(limits.completion_s) as cm:
                    resp = await self._router.acompletion(
                        model=tier.value, messages=messages, stream=False, **opts
                    )
                    a.reached_provider = True
            except litellm.Timeout as exc:
                raise CompletionTimeout(
                    "router timed out waiting for completion"
                ) from exc
            except TimeoutError as exc:
                if cm.expired():
                    raise CompletionTimeout(
                        "router completion request timed out"
                    ) from exc
                raise

            a.chunks += 1
            a.served = getattr(resp, "model", None)
            a.usage = getattr(resp, "usage", None)

            if isinstance(resp, ModelResponse):
                a.parts.append(resp.choices[0].message.content or "")
            a.outcome = Outcome.OK

        except Exception as exc:
            _classify(a, exc)
        finally:
            record = self._close(a, ctx, tier, messages)

        if a.cause is not None:
            raise self._failure(a, record) from a.cause
        return Completion(text="".join(a.parts), record=record)

    # ---------- internals ----------
    @staticmethod
    def _with_turn(ctx: CallContext):
        return ctx if ctx.turn_id else replace(ctx, turn_id=uuid7().hex)

    @staticmethod
    def _failure(a: _Attempt, record: CallRecord) -> AttemptFailure:
        assert a.failure_kind is not None
        return AttemptFailure(
            kind=a.failure_kind,
            record=record,
            partial_text="".join(a.parts),
            tool_call_seen=a.tool_call_seen,
        )

    def _close(
        self, a: _Attempt, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]]
    ) -> CallRecord:
        assert ctx.turn_id is not None
        prompt, completion, estimated = self._tokens(a, tier, messages)
        rec = CallRecord(
            call_id=uuid7().hex,
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            purpose=ctx.purpose,
            cluster=self._cluster,
            requested_model=tier.value,
            served_model=a.served,
            prompt_tokens=prompt,
            completion_tokens=completion,
            latency_s=time.perf_counter() - a.t0,
            ttfc_s=a.ttfc_s,
            cost_scaled=self._cost(a.served, prompt, completion),
            estimated=estimated,
            outcome=a.outcome,
            started_at=a.started_at,
            turn_id=ctx.turn_id,
            attempt_index=ctx.attempt_index,
            chunks_received=a.chunks,
            failure_kind=a.failure_kind,
            scale=CostConfig.SCALE,
        )

        self._ledger.record(rec)
        return rec

    @staticmethod
    def _tokens(a: _Attempt, tier: Tier, messages: list[dict]) -> tuple[int, int, bool]:
        if not a.reached_provider:
            return 0, 0, False

        if a.usage is not None:
            # Safely navigate usage dictionary or object properties
            p_tok = getattr(a.usage, "prompt_tokens", None) or getattr(
                getattr(a.usage, "prompt", None), "tokens", 0
            )
            c_tok = getattr(a.usage, "completion_tokens", None) or getattr(
                getattr(a.usage, "completion", None), "tokens", 0
            )
            return int(p_tok), int(c_tok), False

        model = a.served or tier.value
        try:
            return (
                token_counter(model=model, messages=messages),
                token_counter(model=model, text="".join(a.parts)),
                True,
            )
        except Exception as exc:
            log.exception("token estimate failed: %s", exc)
            return 0, 0, True

    def _cost(self, served: str | None, prompt: int, completion: int) -> int:
        if served is None or (prompt == 0 and completion == 0):
            return 0
        try:
            return self._pricing.cost_scaled(served, prompt, completion)
        except UnknownModelError:
            log.error("no price for served model %r; recording cost 0", served)
            return 0
