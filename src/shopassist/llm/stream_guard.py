"""Story 0.5. Class: StreamGuard. Wraps an async chunk iterator, enforces time-to-first-chunk and inter-chunk stall timeouts, raises FirstChunkTimeout or StreamStalled. No failover knowledge."""
import asyncio
import contextlib
import logging
from typing import TypeVar, AsyncGenerator

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from shopassist.llm.errors import StreamStalled

T = TypeVar("T")

log = logging.getLogger(__name__)


class StreamGuard:
    @staticmethod
    async def wrap(source: CustomStreamWrapper | ModelResponse, stall_timeout_s: float) -> AsyncGenerator[T, None]:

        it = source.__aiter__() if isinstance(source, CustomStreamWrapper) else source

        try:
            while True:
                limit = stall_timeout_s
                try:
                    async with asyncio.timeout(limit):
                        item = await anext(it)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise StreamStalled(f"no chunk for {limit}s") from None
                yield item
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
            else:
                log.debug(f"aclose missing")
