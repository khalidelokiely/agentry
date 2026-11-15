"""Story 0.3. Classes: ChatRequest, SSE Event payloads, and wire formatter."""
from pydantic import BaseModel, ConfigDict, Field

from shopassist.llm.models import Tier


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(..., min_length=1)
    tier: Tier = Field(default=Tier.FAST)

class DeltaEvent(BaseModel):
    text: str


class DoneEvent(BaseModel):
    reply_length: int
    prompt_tokens: int
    completion_tokens: int
    cost_scaled: int
    served_model: str | None = None


class ErrorEvent(BaseModel):
    kind: str
    chunks_received: int


def format_sse(event: str, payload: BaseModel) -> str:
    return f"event: {event}\ndata: {payload.model_dump_json()}\n\n"