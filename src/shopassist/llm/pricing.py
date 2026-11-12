"""Story 0.1. Class: PricingTable. Per-model USD per 1M input/output tokens. cost_scaled(model, prompt_tokens, completion_tokens)."""
from dataclasses import dataclass

from shopassist.llm.cost import CostConfig


class UnknownModelError(KeyError):
    """Raised when no price is registered for a served model."""


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_scaled_per_token: int
    output_scaled_per_token: int

    @classmethod
    def from_usd_per_million(cls, input_usd: str, output_usd: str) -> "ModelPrice":
        return cls(
            input_scaled_per_token=CostConfig.scaled_per_token(input_usd),
            output_scaled_per_token=CostConfig.scaled_per_token(output_usd),
        )


class PricingTable:
    """Maps the SERVED model string (not the tier alias) to its price."""

    def __init__(self) -> None:
        self._prices = {}

    @staticmethod
    def _key(model: str) -> str:
        return model.split("/", 1)[-1]  # "gemini/x" and "x" -> "x"

    def register(self, model: str, input_usd: str, output_usd: str) -> None:
        key = self._key(model)
        new = ModelPrice.from_usd_per_million(input_usd, output_usd)
        old = self._prices.get(key)
        if old is not None and old != new:
            raise ValueError(f"conflicting price for {model!r}")
        self._prices[key] = new

    def has(self, model: str) -> bool:
        return self._key(model) in self._prices

    def price_for(self, model: str) -> ModelPrice:
        try:
            return self._prices[self._key(model)]
        except KeyError:
            raise UnknownModelError(f"no price registered for model {model!r}") from None

    def cost_scaled(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> int:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        p = self.price_for(model)
        return CostConfig.compute(
            prompt_tokens,
            completion_tokens,
            p.input_scaled_per_token,
            p.output_scaled_per_token,
        )

pricing = PricingTable()