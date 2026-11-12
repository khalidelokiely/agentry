"""FastAPI dependencies so tests can swap fakes. Functions: get_llm_client(), get_session_store()."""
from typing import Annotated

from fastapi import Request, Depends

from shopassist.llm.gateway import LLMGateway
from shopassist.llm.usage import UsageLedger


def get_llm_gateway(request: Request) -> LLMGateway:
    return request.app.state.gateway


def get_ledger(request: Request) -> UsageLedger:
    return request.app.state.ledger


GatewayDependency = Annotated[LLMGateway, Depends(get_llm_gateway)]
LedgerDependency = Annotated[UsageLedger, Depends(get_ledger)]
