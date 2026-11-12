"""Story 0.1. Classes: CallRecord
(call_id, session_id, purpose, requested_model, served_model, prompt_tokens, completion_tokens, latency, ttfc, cost, estimated, outcome)
and UsageLedger (record, by_session, totals_by_purpose)."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from shopassist.llm.cost import CostConfig


class Purpose(StrEnum):
    CHAT = "chat"
    SUMMARIZER = "summarizer"


class Outcome(StrEnum):
    OK = "ok"
    STALLED = "stalled"
    FIRST_CHUNK_TIMEOUT = "first_chunk_timeout"
    CANCELLED = "cancelled"
    ERROR = "error"
    AUTHENTICATION_ERROR = "authentication_error"
    CONNECT_TIMEOUT = "connect_timeout"
    TIMEOUT = "timeout"


class FailureKind(StrEnum):
    OVERLOADED = "overloaded"  # 429 / 503, provider rejected the request
    FIRST_CHUNK_TIMEOUT = "first_chunk_timeout"
    STALLED = "stalled"
    ERROR = "error"
    AUTHENTICATION_ERROR = "authentication_error"
    CONNECT_TIMEOUT = "connect_timeout"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class CallRecord:
    call_id: str
    turn_id: str
    session_id: str | None
    user_id: str | None
    purpose: Purpose
    cluster: str | None
    requested_model: str
    served_model: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ttfc_s: float | None
    cost_scaled: int
    estimated: bool
    outcome: Outcome
    started_at: datetime
    attempt_index: int
    chunks_received: int
    failure_kind: FailureKind | None
    scale: int = CostConfig.SCALE


@dataclass(slots=True)
class _Acc:  # private, mutable, never leaves the method
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_calls: int = 0
    cost_scaled: int = 0


@dataclass(frozen=True, slots=True)
class Totals:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_calls: int
    cost_scaled: int


class UsageLedger:
    _records: list[CallRecord]

    def __init__(self) -> None:
        self._records = []

    def record(self, rec: CallRecord) -> None:
        self._records.append(rec)

    def by_session(self, session_id: str) -> list[CallRecord]:
        out: list[CallRecord] = []
        for record in self._records:
            if record.session_id == session_id:
                out.append(record)
        return out

    def totals_by_purpose(
            self, session_id: str | None = None
    ) -> dict[Purpose, Totals]:
        acc: dict[Purpose, _Acc] = defaultdict(_Acc)

        for r in self._records:
            if session_id is not None and r.session_id != session_id:
                continue
            a = acc[r.purpose]
            a.calls += 1
            a.prompt_tokens += r.prompt_tokens
            a.completion_tokens += r.completion_tokens
            a.estimated_calls += int(r.estimated)
            a.cost_scaled += CostConfig.rescale(r.cost_scaled, r.scale, CostConfig.SCALE)

        return {
            p: Totals(
                calls=a.calls,
                prompt_tokens=a.prompt_tokens,
                completion_tokens=a.completion_tokens,
                estimated_calls=a.estimated_calls,
                cost_scaled=a.cost_scaled,
            )
            for p, a in acc.items()
        }

    @property
    def records(self):
        return self._records
