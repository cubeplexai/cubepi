"""Extra coverage for cubepi.tracing.recorder — pins the contracts
codex flagged in the round-1 review and exercises the response-body
shape parsers and error-type derivation paths that aren't covered by
the FauxProvider end-to-end tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from cubepi.agent.agent import Agent
from cubepi.providers.base import Model
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer


MODEL = Model(id="faux-1", provider="faux")


class _Capture(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# ---------------------------------------------------------------------------
# P2-1: every span must carry cubepi.run_id (codex round 1)
# ---------------------------------------------------------------------------


class TestRunIdOnEverySpan:
    async def test_run_id_propagated_to_all_spans(self):
        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = _Capture()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        assert exporter.spans, "expected spans"
        run_ids = {_attrs(s).get("cubepi.run_id") for s in exporter.spans}
        assert len(run_ids) == 1
        assert None not in run_ids
        # Same run id appears on invoke_agent / cubepi.turn / chat ...
        assert run_ids.pop() is not None


class TestJsonlSharding:
    async def test_jsonl_shards_all_spans_under_run_file(self, tmp_path):
        """All spans from one run must end up in the same per-run file —
        codex flagged that without cubepi.run_id on child spans the
        JsonlSpanExporter routed them to unknown-run.jsonl."""
        from cubepi.tracing.exporters import JsonlSpanExporter

        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = JsonlSpanExporter(directory=tmp_path)
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        files = list(tmp_path.rglob("*.jsonl"))
        assert len(files) == 1, f"expected one file, got {files}"
        assert "unknown-run" not in files[0].name


# ---------------------------------------------------------------------------
# P2-2: cooperative abort path on chat span (codex round 1)
# ---------------------------------------------------------------------------


class TestChatSpanAbort:
    async def test_chat_span_marks_aborted_on_cooperative_abort(self):
        provider = FauxProvider(tokens_per_second=10.0)
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = _Capture()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        tracer.attach(agent)
        provider.append_responses([faux_assistant_message("x" * 600)])

        run = asyncio.create_task(agent.prompt("hi"))
        await asyncio.sleep(0.1)
        agent.abort()
        await run
        await tracer.shutdown()

        chats = [s for s in exporter.spans if s.name.startswith("chat ")]
        assert len(chats) == 1
        attrs = _attrs(chats[0])
        # Cooperative abort: no exception, but body has stop_reason="aborted".
        assert attrs.get("cubepi.aborted") is True
        assert attrs.get("error.type") == "cubepi.aborted"
        # Status remains UNSET — abort is a control signal, not a failure.
        assert chats[0].status.status_code == StatusCode.UNSET


# ---------------------------------------------------------------------------
# Response body shape parsing — direct unit tests against Recorder helpers
# ---------------------------------------------------------------------------


class _Span:
    """Lightweight stand-in for an OTel Span — just records set_attribute."""

    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


def _recorder():
    from cubepi.tracing.recorder import Recorder

    tracer = Tracer(service_name="t", exporters=[])
    return Recorder(tracer)


class TestAnthropicResponseShape:
    def test_anthropic_usage_reconciles_cache_tokens(self):
        rec = _recorder()
        span = _Span()
        body = {
            "stop_reason": "end_turn",
            "model": "claude-test",
            "id": "msg_1",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 10,
            },
        }
        rec._record_chat_response_attrs(span, body)
        # input_tokens = 100 + 30 + 10
        assert span.attrs["gen_ai.usage.input_tokens"] == 140
        assert span.attrs["gen_ai.usage.output_tokens"] == 50
        assert span.attrs["gen_ai.usage.cache_read.input_tokens"] == 30
        assert span.attrs["gen_ai.usage.cache_creation.input_tokens"] == 10
        assert span.attrs["gen_ai.response.finish_reasons"] == ["end_turn"]
        assert span.attrs["gen_ai.response.model"] == "claude-test"
        assert span.attrs["gen_ai.response.id"] == "msg_1"


class TestOpenAIChatResponseShape:
    def test_openai_chat_usage_and_finish_reason(self):
        rec = _recorder()
        span = _Span()
        body = {
            "id": "chatcmpl-1",
            "model": "gpt-test",
            "object": "chat.completion",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 75,
                "completion_tokens": 25,
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        }
        rec._record_chat_response_attrs(span, body)
        assert span.attrs["gen_ai.usage.input_tokens"] == 75
        assert span.attrs["gen_ai.usage.output_tokens"] == 25
        assert span.attrs["gen_ai.usage.cache_read.input_tokens"] == 5
        assert span.attrs["gen_ai.response.finish_reasons"] == ["stop"]
        assert span.attrs["gen_ai.response.id"] == "chatcmpl-1"
        assert span.attrs["gen_ai.response.model"] == "gpt-test"


class TestOpenAIResponsesShape:
    def test_openai_responses_reasoning_tokens(self):
        rec = _recorder()
        span = _Span()
        body = {
            "object": "response",
            "id": "resp_1",
            "model": "o4-test",
            "status": "completed",
            "usage": {
                "input_tokens": 60,
                "output_tokens": 40,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens_details": {"reasoning_tokens": 30},
            },
        }
        rec._record_chat_response_attrs(span, body)
        assert span.attrs["gen_ai.usage.input_tokens"] == 60
        assert span.attrs["gen_ai.usage.output_tokens"] == 40
        assert span.attrs["gen_ai.usage.cache_read.input_tokens"] == 20
        assert span.attrs["gen_ai.usage.reasoning.output_tokens"] == 30
        assert span.attrs["gen_ai.response.finish_reasons"] == ["completed"]


# ---------------------------------------------------------------------------
# System-prompt extraction across provider shapes
# ---------------------------------------------------------------------------


class TestSystemPromptExtraction:
    def test_anthropic_string_system(self):
        from cubepi.tracing.recorder import _extract_system_prompt

        assert _extract_system_prompt({"system": "hi"}) == "hi"

    def test_anthropic_cached_system_blocks(self):
        from cubepi.tracing.recorder import _extract_system_prompt

        payload = {
            "system": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        }
        assert _extract_system_prompt(payload) == "helloworld"

    def test_faux_system_prompt_key(self):
        from cubepi.tracing.recorder import _extract_system_prompt

        assert _extract_system_prompt({"system_prompt": "faux"}) == "faux"

    def test_openai_chat_first_system_message(self):
        from cubepi.tracing.recorder import _extract_system_prompt

        payload = {
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ]
        }
        assert _extract_system_prompt(payload) == "be helpful"

    def test_openai_responses_developer_role(self):
        from cubepi.tracing.recorder import _extract_system_prompt

        payload = {
            "input": [
                {"role": "developer", "content": "be precise"},
                {"role": "user", "content": "."},
            ]
        }
        assert _extract_system_prompt(payload) == "be precise"

    def test_no_system_returns_none(self):
        from cubepi.tracing.recorder import _extract_system_prompt

        assert _extract_system_prompt({}) is None
        assert _extract_system_prompt({"messages": []}) is None
        assert _extract_system_prompt({"system": {"unexpected": "shape"}}) is None


# ---------------------------------------------------------------------------
# error.type derivation
# ---------------------------------------------------------------------------


class TestErrorType:
    def test_cancelled_error(self):
        from cubepi.tracing.errors import cubepi_error_type_for

        assert cubepi_error_type_for(asyncio.CancelledError()) == "cubepi.aborted"

    def test_timeout(self):
        from cubepi.tracing.errors import cubepi_error_type_for

        assert cubepi_error_type_for(asyncio.TimeoutError()) == "timeout"
        assert cubepi_error_type_for(TimeoutError()) == "timeout"

    def test_connection_error(self):
        from cubepi.tracing.errors import cubepi_error_type_for

        assert cubepi_error_type_for(ConnectionError()) == "connection_error"

    def test_builtin_exception_uses_qualname(self):
        from cubepi.tracing.errors import cubepi_error_type_for

        assert cubepi_error_type_for(ValueError("x")) == "ValueError"

    def test_module_qualified_class(self):
        from cubepi.tracing.errors import cubepi_error_type_for

        class CustomErr(Exception):
            pass

        err = CustomErr()
        out = cubepi_error_type_for(err)
        assert out.endswith("CustomErr")


# ---------------------------------------------------------------------------
# Schema URL & provider name mapping
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_schema_url_pinned(self):
        from cubepi.tracing.schema import SCHEMA_URL

        assert SCHEMA_URL == "https://opentelemetry.io/schemas/1.41.0"

    def test_provider_name_known(self):
        from cubepi.tracing.schema import map_provider_name

        assert map_provider_name("anthropic") == "anthropic"
        assert map_provider_name("azure_openai") == "azure.ai.openai"
        assert map_provider_name("bedrock") == "aws.bedrock"
        assert map_provider_name("vertex_ai") == "gcp.vertex_ai"

    def test_provider_name_unknown_prefixed(self):
        from cubepi.tracing.schema import map_provider_name

        assert map_provider_name("nobody") == "unknown:nobody"


# ---------------------------------------------------------------------------
# Detach lifecycle
# ---------------------------------------------------------------------------


class TestDetach:
    async def test_detach_unsubscribes(self):
        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("a")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        exporter = _Capture()
        tracer = Tracer(service_name="t", agent_name="a", exporters=[exporter])
        detach = tracer.attach(agent)

        await agent.prompt("first")
        await agent.wait_for_idle()
        await tracer.force_flush()
        first_count = len(exporter.spans)
        assert first_count >= 1

        detach()
        # detach schedules an async unsubscribe task; let it run.
        await asyncio.sleep(0.05)

        provider.append_responses([faux_assistant_message("b")])
        await agent.prompt("second")
        await agent.wait_for_idle()
        await tracer.shutdown()

        # After detach the recorder must no longer produce new spans.
        assert len(exporter.spans) == first_count


# ---------------------------------------------------------------------------
# Tracer config edge cases
# ---------------------------------------------------------------------------


class TestTracerConfig:
    def test_explicit_resource_passes_through(self):
        from opentelemetry.sdk.resources import Resource

        custom = Resource.create(
            {"service.name": "explicit"},
            schema_url="https://opentelemetry.io/schemas/1.41.0",
        )
        tracer = Tracer(resource=custom, exporters=[])
        assert tracer.resource is custom

    async def test_async_context_manager_calls_shutdown(self):
        exporter = _Capture()
        async with Tracer(service_name="t", exporters=[exporter]) as t:
            assert t.resource is not None
        # No raise — shutdown ran via __aexit__.
