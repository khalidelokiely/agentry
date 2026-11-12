from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from shopassist.api.routes.healthz import router
from shopassist.llm.gateway import LLMGateway
from shopassist.llm.usage import UsageLedger


def create_lifespan(gateway: LLMGateway, ledger: UsageLedger):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.gateway = gateway
        app.state.ledger = ledger
        try:
            yield
        finally:
            app.state.ledger = None

    return lifespan


def create_app(gateway: LLMGateway, ledger: UsageLedger) -> FastAPI:
    app = FastAPI(lifespan=create_lifespan(gateway, ledger))
    app.include_router(router)
    return app
