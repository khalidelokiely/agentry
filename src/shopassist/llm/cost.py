from decimal import Decimal


class CostConfig:
    """Single source of truth for money math.

    All costs are integers in units of 1/SCALE USD.
    Change SCALE here and nowhere else.
    """

    SCALE = 1_000_000_000              # 1 USD == SCALE scaled units
    TOKENS_PER_PRICE_UNIT = 1_000_000  # providers quote USD per 1M tokens

    @classmethod
    def usd_to_scaled(cls, usd: float | str | Decimal) -> int:
        return round(Decimal(str(usd)) * cls.SCALE)

    @classmethod
    def scaled_to_usd(cls, scaled: int) -> float:
        # display/reporting only, never store the result
        return scaled / cls.SCALE

    @classmethod
    def scaled_per_token(cls, usd_per_million: float | str | Decimal) -> int:
        return round(
            Decimal(str(usd_per_million)) * cls.SCALE / cls.TOKENS_PER_PRICE_UNIT
        )

    @staticmethod
    def compute(
        prompt_tokens: int,
        completion_tokens: int,
        input_scaled_per_token: int,
        output_scaled_per_token: int,
    ) -> int:
        return (
            prompt_tokens * input_scaled_per_token
            + completion_tokens * output_scaled_per_token
        )

    @staticmethod
    def rescale(value: int, from_scale: int, to_scale: int) -> int:
        if from_scale == to_scale:
            return value
        return round(Decimal(value) * to_scale / from_scale)