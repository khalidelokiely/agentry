"""ChatCLI scratch loop. Commands: /new (new session), /cut N (stop reading after N chars, 0 = off), /quit."""
import asyncio
import logging
import os
from contextlib import aclosing
from uuid import uuid7

from InquirerPy import inquirer
from rich import print

from shopassist.llm.context import CallContext
from shopassist.llm.cost import CostConfig
from shopassist.llm.errors import AttemptFailure
from shopassist.llm.events import TextDelta, Completion
from shopassist.llm.gateway import LLMGateway
from shopassist.llm.models import Tier
from shopassist.llm.usage import Purpose, UsageLedger
from shopassist.memory.compactor import Compactor
from shopassist.memory.memory import Memory
from shopassist.settings import Settings
from shopassist.wiring import build_gateway

log = logging.getLogger(__name__)


def show(ledger: UsageLedger, session_id: str) -> None:
    rows = ledger.by_session(session_id)
    if rows:
        print(rows[-1])
    for purpose, t in ledger.totals_by_purpose(session_id).items():
        usd = CostConfig.scaled_to_usd(t.cost_scaled)
        print(f"{purpose.value}: {t.calls} calls, {t.prompt_tokens} in / {t.completion_tokens} out, "
              f"{t.estimated_calls} estimated, ${usd:.6f}")


def pick_tier() -> str:
    return inquirer.select(message="Tier:", choices=[t.value for t in Tier]).execute()


async def main() -> None:
    bouncer, ledger = build_gateway(settings=Settings(), primary_provider='gemini')
    memory = Memory(max_window=2)

    compaction_sema = asyncio.Semaphore(1)
    background_tasks = set()

    session_id = uuid7().hex
    cut_after = 0
    while True:
        try:
            prompt = (await asyncio.to_thread(input, "\nPrompt (/new, /cut N, /quit): ")).strip()
            prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/new":
            memory.clear()
            session_id = uuid7().hex
            print("new session", session_id)
            continue
        if prompt.startswith("/cut"):
            arg = prompt[4:].strip()
            cut_after = int(arg) if arg.isdigit() else 0
            print(f"cut_after={cut_after}")
            continue

        tier = Tier(await asyncio.to_thread(pick_tier))
        ctx = CallContext(purpose=Purpose.CHAT, session_id=session_id)

        memory.append(role="user", content=prompt)

        try:
            opts = {}
            if tier.value in ["smart", "thinking"]:
                opts["reasoning_effort"] = "low"

            print(memory.get_messages())
            async with aclosing(
                    bouncer.stream(ctx, tier, memory.get_messages(), **opts)) as events:
                seen = 0
                async for ev in events:
                    if isinstance(ev, TextDelta):
                        print(ev.text, end="", flush=True)
                        seen += len(ev.text)
                        if cut_after and seen >= cut_after:
                            break
                if isinstance(ev, Completion):
                    memory.append(role="assistant", content=ev.text)

                if memory.needs_compaction() and not compaction_sema.locked():
                    compaction_task = asyncio.create_task(run_compaction(
                        sema=compaction_sema,
                        gateway=bouncer,
                        memory=memory,
                        ctx=ctx,
                        keep_last=1
                    ))
                    background_tasks.add(compaction_task)
                    compaction_task.add_done_callback(background_tasks.discard)
            print()
        except AttemptFailure as f:
            print(f"\n[{f.kind.value}] after {f.chunks_received} chunks, "
                  f"{len(f.partial_text)} chars of partial text")
        show(ledger, session_id)


async def plan_summarize_commit(memory: Memory, gateway: LLMGateway, ctx: CallContext, tier: Tier,
                                keep_last: int = 4) -> None:
    log.info("updating memory...")
    plan = Compactor.plan(memory, keep_last=keep_last)

    if plan is None:
        log.info("compaction planning failed")
        return

    result = await Compactor.summarize(gateway=gateway, ctx=ctx, plan=plan, tier=tier)

    if result is None:
        log.info("summarizing failed")
        return

    commit = memory.commit(result)
    if not commit:
        log.info("memory commit failed")


async def run_compaction(sema: asyncio.Semaphore, memory: Memory, gateway: LLMGateway, ctx: CallContext,
                         keep_last: int = 4) -> None:
    async with sema:
        await plan_summarize_commit(memory, gateway, ctx, Tier.FLAKY, keep_last)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        os._exit(130)  # dev tool: don't wait for a thread blocked in input()
