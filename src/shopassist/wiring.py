import logging
from typing import Sequence

from shopassist.llm.bouncer import Bouncer
from shopassist.llm.client import LLMClient
from shopassist.llm.gateway import LLMGateway
from shopassist.llm.models import ModelPricing
from shopassist.llm.pricing import PricingTable
from shopassist.llm.usage import UsageLedger
from shopassist.router_factory import RouterFactory, ClusterRuntime
from shopassist.settings import Settings

log = logging.getLogger(__name__)


def build_gateway(
        settings: Settings,
        primary_provider: str,
        secondary_providers: Sequence[str] = (),
        local_providers: Sequence[str] = (),
        skip_invalid_providers=False) -> tuple[LLMGateway, UsageLedger]:
    if primary_provider in secondary_providers or primary_provider in local_providers:
        raise ValueError(f"Primary provider {primary_provider} found in secondary|local provider list.")

    factory = RouterFactory(settings)
    ledger = UsageLedger()
    bouncer = Bouncer(
        primary=_build_client(primary_provider, factory.build(primary_provider), ledger),
        secondary=_build(
            providers=secondary_providers,
            factory=factory,
            ledger=ledger,
            skip_invalid_providers=skip_invalid_providers),
        local=_build(
            providers=local_providers,
            factory=factory,
            ledger=ledger,
            skip_invalid_providers=skip_invalid_providers),
    )

    return bouncer, ledger


def _register_pricing(pricing: PricingTable, model_pricing: dict[str, ModelPricing]):
    for m, p in model_pricing.items():
        pricing.register(m, p.input_usd, p.output_usd)


def _build_client(cluster: str, runtime: ClusterRuntime, ledger: UsageLedger) -> LLMGateway:
    pricing = PricingTable()
    _register_pricing(pricing, runtime.model_pricing)

    client = LLMClient(
        cluster=cluster,
        router=runtime.router,
        ledger=ledger,
        pricing=pricing,
        timeouts=runtime.timeouts,
    )

    return client


def _build(providers: Sequence[str], factory: RouterFactory, ledger: UsageLedger,
           skip_invalid_providers: bool = False) -> list[LLMGateway]:
    llm_clients = []

    for p in providers:
        if factory.exists(p):
            llm_clients.append(_build_client(p, factory.build(p), ledger=ledger))
        elif skip_invalid_providers:
            log.warning("Skipping invalid provider: %s", p)
        else:
            raise ValueError(f"Invalid provider in provider list: {p}")

    return llm_clients
