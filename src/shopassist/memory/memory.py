"""Class: Memory. Holds system prompt, facts, summary, window."""
import json
from collections.abc import Sequence
from typing import Any

from shopassist.memory.messages import CompactionResult, Message

SYSTEM_MEMORY = """
Known facts (authoritative):
{facts}
Conversation so far:
{summary}
"""


class Memory:
    def __init__(self, max_window: int = 12):
        self._messages: list[Message] = []
        self._max_window: int = max_window
        self._summary: str | None = None
        self._facts: dict[str, Any] = {}
        self._version: int = 0

    def append(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content)
        self._messages.append(message)
        return message

    def view(self) -> list[Message]:
        return list(self._messages)

    def needs_compaction(self) -> bool:
        return len(self._messages) > self._max_window

    def get_summary(self) -> str | None:
        return self._summary

    def get_facts(self) -> dict[str, Any]:
        return dict(self._facts)

    def get_version(self) -> int:
        return self._version

    def get_messages(self) -> list[dict[str, str]]:
        system_message = None

        if self._summary is not None or len(self._facts) > 0:
            system_message = Message(
                role="system",
                content=SYSTEM_MEMORY.format(
                    facts=json.dumps(self._facts),
                    summary=json.dumps(self._summary)
                )
            )

        return [
            {"role": m.role, "content": m.content}
            for m in
            (
                [system_message] + self._messages if system_message is not None else self._messages
            )
        ]

    def commit(self, result: CompactionResult) -> bool:
        if result.plan.version != self._version:
            return False

        purge_ids = set(result.plan.purge_ids)

        new_messages = []
        purged_messages = []

        for message in self._messages:
            if message.id in purge_ids:
                purge_ids.remove(message.id)
                purged_messages.append(message)
            else:
                new_messages.append(message)

        if len(purge_ids) != 0:
            return False

        self._facts = {
            k: v
            for k, v in (self._facts | result.facts).items()
            if v is not None
        }
        self._summary = result.summary
        self._messages = new_messages
        self._archive(purged_messages)

        self._version += 1
        return True

    def _archive(self, messages: Sequence[Message]) -> None:
        pass

    def clear(self) -> None:
        self._messages: list[Message] = []
        self._summary: str | None = None
        self._facts: dict[str, Any] = {}
        self._version += 1
