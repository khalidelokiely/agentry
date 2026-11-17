from typing import Any, AsyncGenerator, Protocol

from shopassist.llm.context import CallContext
from shopassist.llm.events import Completion, LLMEvent
from shopassist.llm.models import Tier


class LLMGateway(Protocol):
    """Satisfied by LLMClient AND Bouncer. Callers depend on this, never on a concrete class."""

    async def stream(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> AsyncGenerator[LLMEvent]: ...

    async def complete(
        self, ctx: CallContext, tier: Tier, messages: list[dict[str, Any]], **opts: Any
    ) -> Completion: ...
