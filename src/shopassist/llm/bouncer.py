from contextlib import aclosing
from dataclasses import replace
from typing import Any, AsyncGenerator
from uuid import uuid7

from shopassist.llm.context import CallContext
from shopassist.llm.events import Completion
from shopassist.llm.gateway import LLMEvent, LLMGateway
from shopassist.llm.models import Tier


class Bouncer:
    """Today: mint the turn, call one gateway, let failures pass through (the dummy tunnel).
    Later: hold several gateways, catch AttemptFailure, pick who continues."""

    def __init__(self, primary: LLMGateway, secondary=None, local=None) -> None:
        if local is None:
            local = []
        if secondary is None:
            secondary = []
        self._primary = primary
        self._secondary = secondary
        self._local = local

    @staticmethod
    def _start_turn(ctx: CallContext) -> CallContext:
        return replace(ctx, turn_id=ctx.turn_id or uuid7().hex, attempt_index=0)

    async def stream(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> AsyncGenerator[LLMEvent]:
        ctx = self._start_turn(ctx)

        response_stream = self._primary.stream(ctx, tier, messages, **opts)

        async with aclosing(response_stream) as stream:
            async for event in stream:
                yield event

    async def complete(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> Completion:
        ctx = self._start_turn(ctx)
        return await self._primary.complete(ctx, tier, messages, **opts)
