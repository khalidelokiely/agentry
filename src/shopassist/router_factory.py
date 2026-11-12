from dataclasses import dataclass

from litellm import Router

from shopassist.llm.models import Tier, Timeouts, ModelPricing
from shopassist.settings import Settings


@dataclass(frozen=True)
class ClusterRuntime:
    router: Router
    model_pricing: dict[str, ModelPricing]
    timeouts: dict[Tier, Timeouts]
    tier_models: dict[Tier, str]


class RouterFactory:
    def __init__(self, settings: Settings):
        self._settings = settings

    def exists(self, provider: str) -> bool:
        return provider in self._settings.models

    def build(self, provider: str, **kwargs) -> ClusterRuntime:
        if provider not in self._settings.models:
            raise ValueError(f"Unknown provider {provider}")

        provider_settings = self._settings.models.get(provider) or {}
        provider_options = self._settings.options.get(provider) or {}
        all_fallbacks = self._settings.model_fallbacks()

        kwargs_models = kwargs.pop("model_list", None) or []
        kwargs_fallbacks = kwargs.pop("fallbacks", None) or []
        kwargs_retries = kwargs.pop("num_retries", None)

        merged_models = [m.spec.get_config() for m in provider_settings.values()] + kwargs_models

        base_fallbacks = all_fallbacks.get(provider, [])
        merged_fallbacks = base_fallbacks + kwargs_fallbacks

        if kwargs_retries is not None:
            merged_retries = kwargs_retries
        elif provider_options.get("num_retries") is not None:
            merged_retries = provider_options["num_retries"]
        else:
            merged_retries = self._settings.num_retries

        litellm_router = Router(
            model_list=merged_models,
            fallbacks=merged_fallbacks,
            num_retries=merged_retries,
            **kwargs
        )

        provider_timeouts = {
            tier: self._settings.timeouts_for(provider, tier)
            for tier in provider_settings.keys()
        }

        runtime = ClusterRuntime(
            router=litellm_router,
            model_pricing={m.name: m.pricing for m in provider_settings.values()},
            timeouts=provider_timeouts,
            tier_models={t: m.name for t, m in provider_settings.items()},
        )

        return runtime
