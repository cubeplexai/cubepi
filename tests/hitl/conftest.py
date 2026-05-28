# tests/hitl/conftest.py
import asyncio


async def await_pending(channel, *, timeout: float = 2.0) -> None:
    """Wait until channel.pending becomes non-None, or fail the test on timeout.

    Use this in tests that race a host coroutine against an awaiting agent.
    Replaces the `while ch.pending is None: await asyncio.sleep(0)` pattern,
    which silently hangs if the host task crashes.
    """

    async def _wait():
        while channel.pending is None:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(_wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise AssertionError(
            f"channel.pending did not become set within {timeout}s"
        ) from exc
