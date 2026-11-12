from dataclasses import dataclass
from shopassist.llm.usage import Purpose


@dataclass(frozen=True, slots=True)
class CallContext:
    """Who/why for one call. Grows (tenant, trace id) without touching signatures."""
    purpose: Purpose
    session_id: str | None = None
    user_id: str | None = None
    turn_id: str | None = None      # minted by the Bouncer
    attempt_index: int = 0          # set by the Bouncer