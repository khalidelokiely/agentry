"""Story 0.2. Class: Message (id, role, content, ts). Stable ids are what fix the compaction race."""
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid7


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    id: str = field(default_factory=lambda: uuid7().hex)

    def to_llm(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    version: int
    purge_ids: tuple[str, ...]
    purge_messages: tuple[Message, ...]
    summary: str | None
    facts: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactionResult:
    plan: CompactionPlan
    summary: str
    facts: dict[str, Any]


class CompactionArtifact:
    pass
