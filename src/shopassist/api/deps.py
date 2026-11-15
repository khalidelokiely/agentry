"""FastAPI dependencies so tests can swap fakes."""
import asyncio
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid7

from fastapi import Depends, Header, Request

from shopassist.llm.gateway import LLMGateway
from shopassist.llm.usage import UsageLedger
from shopassist.memory.memory import Memory


def get_llm_gateway(request: Request) -> LLMGateway:
    return request.app.state.gateway

def get_ledger(request: Request) -> UsageLedger:
    return request.app.state.ledger

GatewayDependency = Annotated[LLMGateway, Depends(get_llm_gateway)]
LedgerDependency = Annotated[UsageLedger, Depends(get_ledger)]


@dataclass(slots=True)
class Session:
    session_id: str
    memory: Memory = field(default_factory=lambda: Memory(max_window=12))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class SessionStore:
    def __init__(self, max_window: int = 12) -> None:
        self._max_window: int = max_window
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None) -> Session:
        sid = session_id.strip() if session_id and session_id.strip() else uuid7().hex
        if sid not in self._sessions:
            self._sessions[sid] = Session(
                session_id=sid,
                memory=Memory(max_window=self._max_window)
            )
        return self._sessions[sid]

def resolve_session(
    request: Request,
    x_session_id: Annotated[str | None, Header()] = None,
) -> Session:
    return request.app.state.sessions.get_or_create(x_session_id)

SessionDependency = Annotated[Session, Depends(resolve_session)]