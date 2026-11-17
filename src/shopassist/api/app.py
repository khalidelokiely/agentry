from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from shopassist.api.deps import SessionStore
from shopassist.api.routes.healthz import router as healthz_router
from shopassist.api.routes_chat import router as chat_router
from shopassist.llm.gateway import LLMGateway
from shopassist.llm.usage import UsageLedger


def create_lifespan(gateway: LLMGateway, ledger: UsageLedger):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.gateway = gateway
        app.state.ledger = ledger
        app.state.sessions = SessionStore()
        try:
            yield
        finally:
            app.state.ledger = None
            app.state.sessions = None
            app.state.gateway = None

    return lifespan


def create_app(gateway: LLMGateway, ledger: UsageLedger) -> FastAPI:
    app = FastAPI(title="ShopAssist", lifespan=create_lifespan(gateway, ledger))
    app.include_router(healthz_router)
    app.include_router(chat_router)
    return app
