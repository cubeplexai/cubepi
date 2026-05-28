"""Shared test helpers for HITL test suite.

`await_pending` is intentionally a plain async helper (not a pytest fixture)
because downstream tests call it directly with the channel as a positional
argument, e.g. `await await_pending(ch)`. Promoting it to a fixture would
force every test that uses it to take it as a parameter and then invoke
it — more boilerplate for no benefit.
"""

import asyncio


async def await_pending(channel, *, timeout: float = 2.0) -> None:
    """Wait until channel.pending becomes non-None, or fail the test on timeout.

    Replaces the `while ch.pending is None: await asyncio.sleep(0)` pattern,
    which silently hangs if the host coroutine crashes.
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
