"""Classes: LiteLLMParams, ModelSpec (dataclasses for the router model list) and Tier(StrEnum) for fast|smart|thinking|flaky."""
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import StrEnum
from typing import Any, Optional


class Tier(StrEnum):
    FAST = "fast"
    SMART = "smart"
    THINKING = "thinking"
    FLAKY = "flaky"

@dataclass(frozen=True, slots=True)
class LiteLLMParams:
    model: str
    api_key: Optional[str]= field(default=None, repr=False)
    api_base: Optional[str] = None
    timeout: Optional[float] = 30
    stream_timeout: Optional[float] = 10


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_name: Tier
    litellm_params: LiteLLMParams
    rpm: Optional[int] = None
    tpm: Optional[int] = None

    def get_config(self) -> dict:
        params = {k: v for k, v in asdict(self.litellm_params).items() if v is not None}
        d: dict[str, str | dict[str, Any] | int | None] = {
            "model_name": self.model_name.value,
            "litellm_params": params
        }

        if self.rpm is not None:
            d["rpm"] = self.rpm
        if self.tpm is not None:
            d["tpm"] = self.tpm
        return d

"""
ModelPricing carries pricing metadata for the model
"""
@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_usd: str   # USD per 1M tokens, as printed on the pricing page
    output_usd: str

    def __post_init__(self) -> None:
        Decimal(self.input_usd)
        Decimal(self.output_usd)

@dataclass(frozen=True, slots=True)
class Timeouts:
    stream_open_s: float   # whole fallback chain, until the first chunk arrives
    completion_s: float    # whole fallback chain, until a non-streaming reply is complete
    stall_s: float         # gap between chunks, enforced by the guard

"""
ModelEntry is the record used to configure a fully qualified usable model into litellm
"""
@dataclass(frozen=True, slots=True)
class ModelEntry:
    spec: ModelSpec
    pricing: ModelPricing
    stall_timeout_s: float | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("stall_timeout_s", self.stall_timeout_s),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive")
        object.__setattr__(self, "name", self.spec.litellm_params.model)