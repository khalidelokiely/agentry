import os
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from shopassist.llm.models import ModelEntry, Tier, Timeouts

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = Path(os.environ.get("SHOPASSIST_CONFIG", _ROOT / "config.toml"))


class Settings(BaseSettings):
    models: dict[str, dict[Tier, ModelEntry]]
    fallbacks: dict[str, dict[Tier, list[Tier]]]
    options: dict[str, dict[str, Any]]

    num_retries: int = Field(2, ge=0)
    request_timeout_s: float = Field(30.0, gt=0)

    stall_timeout_s: float = Field(5.0, gt=0)
    max_window: int = Field(12, gt=0)
    keep_last: int = Field(4, gt=0)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=_ROOT / ".env",
        toml_file=_CONFIG_PATH,
        extra="allow",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="before")
    @classmethod
    def _fill_tier_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for provider, tiers in (data.get("models") or {}).items():
            for tier, entry in tiers.items():
                if not isinstance(entry, dict):
                    continue
                spec = entry.setdefault("spec", {})
                if not isinstance(spec, dict):
                    continue
                declared = spec.setdefault("model_name", tier)
                if str(declared) != str(tier):
                    raise ValueError(
                        f"models.{provider}.{tier}: model_name {declared!r} "
                        f"does not match its table key"
                    )
        return data

    @model_validator(mode="after")
    def _validate_api_keys(self) -> "Settings":
        if self.keep_last >= self.max_window:
            raise ValueError("keep_last must be smaller than max_window")

        resolved_models: dict[str, dict[Tier, ModelEntry]] = {}
        for provider, tiers in self.models.items():
            global_key = self.api_key_for(provider)
            resolved_tiers: dict[Tier, ModelEntry] = {}
            for tier, entry in tiers.items():
                params = entry.spec.litellm_params
                if params.api_key is None:
                    if global_key is None:
                        raise ValueError(
                            f"No API key found for models.{provider}.{tier.value}: "
                            f"set {provider.upper()}_API_KEY in .env"
                        )
                    new_params = replace(params, api_key=global_key)
                    new_spec = replace(entry.spec, litellm_params=new_params)
                    entry = replace(entry, spec=new_spec)
                resolved_tiers[tier] = entry
            resolved_models[provider] = resolved_tiers
        self.models = resolved_models
        return self

    @model_validator(mode="after")
    def _validate_fallbacks(self) -> "Settings":
        for provider, fallback in self.fallbacks.items():
            cluster = self.models.get(provider)
            if cluster is None:
                raise ValueError(f"Provider {provider} not found in configured models.")

            unique_tiers = set()
            for item, goto in fallback.items():
                if item in goto:
                    raise ValueError(
                        f"[CYCLIC FALLBACK @fallbacks.{provider}] "
                        f"Tier {item.value} lists itself as its own fallback."
                    )
                if len(set(goto)) != len(goto):
                    raise ValueError(
                        f"[DUPLICATE FALLBACK @fallbacks.{provider}] "
                        f"Tier {item.value} has duplicate fallback entries."
                    )
                if not goto:
                    raise ValueError(
                        f"[EMPTY FALLBACK @fallbacks.{provider}] "
                        f"Tier {item.value} has no fallback entries."
                    )
                unique_tiers.add(item)
                unique_tiers.update(goto)

            for item in unique_tiers:
                if item not in cluster:
                    raise ValueError(
                        f"Fallback tier {item.value} missing from provider cluster {provider}"
                    )
        return self

    def api_key_for(self, provider: str) -> str | None:
        raw = (self.model_extra or {}).get(f"{provider}_api_key")
        if raw is None:
            return None
        return raw.get_secret_value() if isinstance(raw, SecretStr) else str(raw)

    def _hop_timeout(self, provider: str, tier: Tier, *, stream: bool) -> float:
        params = self.models[provider][tier].spec.litellm_params
        if stream:
            return params.stream_timeout or params.timeout or self.request_timeout_s

        return params.timeout or self.request_timeout_s

    def timeouts_for(self, provider: str, tier: Tier) -> Timeouts:
        entry = self.models[provider][tier]

        # Fetch the retry count for this provider (fallback to global default)
        provider_options = self.options.get(provider, {})
        retries = provider_options.get("num_retries", self.num_retries)
        attempts = 1 + retries

        visited = {tier}
        q = deque([tier])

        while q:
            current = q.popleft()
            for fallback in self.fallbacks.get(provider, {}).get(current, []):
                if fallback not in visited:
                    visited.add(fallback)
                    q.append(fallback)

        completion = sum(self._hop_timeout(provider, t, stream=False) * attempts for t in visited) * 1.2
        stream_open = sum(self._hop_timeout(provider, t, stream=True) * attempts for t in visited) * 1.2

        return Timeouts(
            stall_s=entry.stall_timeout_s
            if entry.stall_timeout_s is not None
            else self.stall_timeout_s,
            completion_s=completion,
            stream_open_s=stream_open,
        )

    def model_fallbacks(self) -> dict[str, list[dict[str, list[str]]]]:
        out: dict[str, list[dict[str, list[str]]]] = {}
        for provider, fallback in self.fallbacks.items():
            out[provider] = []
            for tier, entry in fallback.items():
                out[provider].append({tier.value: [e.value for e in entry]})
        return out