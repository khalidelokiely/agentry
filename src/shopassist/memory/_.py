import json
from dataclasses import dataclass, field

from litellm import LiteLLMUnknownProvider

SUMMARY_PROMPT = """You maintain memory for an ongoing conversation.
Current summary: {summary}
Current facts: {facts}
Older messages to fold in:
====START TRANSCRIPT====
{transcript}
====END TRANSCRIPT======

Return ONLY JSON: {{"summary": "...", "facts": {{...}}}}
- summary: short, factual, updated with the older messages. Keep intent, decisions, commitments, open questions. Drop small talk.
- facts: flat key/value pairs for things that must never be lost (names, IDs, amounts, dates, the user's goal). Newer values overwrite older ones.
- Never invent anything that wasn't said."""


@dataclass
class Memory:
    system: str
    token_usage: int = 0
    last_message_usage: int = 0
    summary: str = ""
    facts: dict = field(default_factory=dict)
    window: list[dict] = field(default_factory=list)
    max_window: int = 12
    keep_last: int = 4

    def messages(self) -> list[dict]:
        ctx = self.system
        if self.facts:
            ctx += "\n\nKnown facts (authoritative):\n" + json.dumps(self.facts)
        if self.summary:
            ctx += "\n\nConversation so far:\n" + self.summary
        return [{"role": "system", "content": ctx}] + self.window

    def add(self, role: str, content: str) -> None:
        self.window.append({"role": role, "content": content})

    def needs_compaction(self) -> bool:
        return len(self.window) > self.max_window

    def update_usage(self, usage: int):
        self.token_usage += usage
        self.last_message_usage = usage

    def compact(self, router) -> None:
        old, recent = self.window[: -self.keep_last], self.window[-self.keep_last:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        try:
            resp = router.completion(
                model="fast",
                messages=[
                    {
                        "role": "user",
                        "content": SUMMARY_PROMPT.format(
                            summary=self.summary,
                            facts=json.dumps(self.facts),
                            transcript=transcript,
                        ),
                    }
                ],
            )
            raw = (
                resp.choices[0]
                .message.content.strip()
                .removeprefix("```json")
                .removesuffix("```")
            )
            data = json.loads(raw)
        except LiteLLMUnknownProvider as e:
            print(f"compaction failed, keeping full window: {e}")
            return  # never drop messages unless the summary succeeded
        self.summary = data["summary"]
        self.facts.update(data.get("facts", {}))
        self.window = recent
