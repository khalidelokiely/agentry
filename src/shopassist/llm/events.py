from dataclasses import dataclass
from shopassist.llm.usage import CallRecord


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class Completion:
    """Full result. Returned by complete(), and yielded as the LAST event of stream()."""
    text: str
    record: CallRecord


LLMEvent = TextDelta | Completion