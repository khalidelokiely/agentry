"""Story 0.5. Exceptions: FirstChunkTimeout, StreamStalled, AllModelsDown. Failover logic keys off these."""
from shopassist.llm.usage import FailureKind, CallRecord

class GuardTimeout(Exception):
    """Base for timeouts raised by StreamGuard"""

class FirstChunkTimeout(GuardTimeout): ...
class CompletionTimeout(GuardTimeout): ...
class ConnectTimeout(GuardTimeout): ...
class StreamStalled(GuardTimeout): ...
class AllModelsDown(GuardTimeout): ...

class AttemptFailure(Exception):
    """The envelope. Raised by LLMClient when an attempt fails. It reports facts, it decides nothing.
    (This is what I called StreamFailure earlier; renamed because complete() raises it too.)"""

    def __init__(
            self,
            kind: FailureKind,
            record: CallRecord,
            partial_text: str = "",
            tool_call_seen: bool = False
    ) -> None:
        super().__init__(f"{kind.value}: attempt {record.call_id}")
        self.kind = kind
        self.record = record
        self.partial_text = partial_text
        self.tool_call_seen = tool_call_seen

    @property
    def chunks_received(self)->int:
        return self.record.chunks_received