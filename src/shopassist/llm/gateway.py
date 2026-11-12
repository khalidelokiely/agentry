from typing import Any, Protocol, AsyncGenerator

from shopassist.llm.context import CallContext
from shopassist.llm.events import Completion, LLMEvent
from shopassist.llm.models import Tier


class LLMGateway(Protocol):
    """Satisfied by LLMClient AND Bouncer. Callers depend on this, never on a concrete class."""

    def stream(
        self, ctx: CallContext, tier: Tier, messages: list[dict], **opts: Any
    ) -> AsyncGenerator[LLMEvent, None]: ...

    async def complete(
        self, ctx: CallContext, tier: Tier, messages: list[dict], **opts: Any
    ) -> Completion: ...