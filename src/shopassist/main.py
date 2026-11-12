"""FastAPI app factory. Function: create_app(). The lifespan builds Router, UsageLedger and LLMClient once and stores them on app.state."""
from shopassist.api.app import create_app
from shopassist.settings import Settings
from shopassist.wiring import build_gateway

gateway, ledger = build_gateway(settings=Settings(), primary_provider='gemini')

app = create_app(gateway=gateway, ledger=ledger)
