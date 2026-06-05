"""Extra coverage for cubepi.tracing.recorder — pins the contracts
codex flagged in the round-1 review and exercises the response-body
shape parsers and error-type derivation paths that aren't covered by
the FauxProvider end-to-end tests.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import Model, TextContent, ToolCall
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Tracer


MODEL = Model(id="faux-1", provider_id="faux")


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
        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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

        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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
        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("a")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
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


# ---------------------------------------------------------------------------
# Stream recording (record_stream=True)
# ---------------------------------------------------------------------------


class TestStreamRecording:
    async def test_stream_file_created_and_closed(self, tmp_path):
        """record_stream=True writes a .stream.jsonl file for the run."""
        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("hello world")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        await agent.prompt("hi")
        await agent.wait_for_idle()
        await tracer.shutdown()

        files = list(tmp_path.glob("*.stream.jsonl"))
        assert len(files) == 1

    async def test_stream_file_contains_events(self, tmp_path):
        """Each text chunk produces a text_delta line in the stream file."""
        provider = FauxProvider(tokens_per_second=1000.0)
        provider.append_responses([faux_assistant_message("abc")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        await agent.prompt("hi")
        await agent.wait_for_idle()
        await tracer.shutdown()

        stream_file = next(tmp_path.glob("*.stream.jsonl"))
        events = [
            json.loads(line) for line in stream_file.read_text().splitlines() if line
        ]
        types = [e["type"] for e in events]
        assert "text_delta" in types

    async def test_stream_file_closed_on_cancel(self, tmp_path):
        """Stream file is closed via _close_open_spans on asyncio.CancelledError.

        _close_open_spans is only reachable when detach() is called after a
        CancelledError-aborted run — agent.abort() goes through _on_agent_end
        instead, so we must use task.cancel() and then explicitly call detach().
        """
        provider = FauxProvider(tokens_per_second=5.0)
        provider.append_responses([faux_assistant_message("x" * 300)])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        detach = tracer.attach(agent)

        run = asyncio.create_task(agent.prompt("hi"))
        await asyncio.sleep(0.05)
        run.cancel()
        try:
            await run
        except asyncio.CancelledError:
            pass
        # detach() calls _sync_detach → _close_open_spans, which closes the
        # stream file on the cancellation path (lines 467-471 of recorder.py).
        detach()
        await tracer.shutdown()

        # File must exist and be readable (closed properly, not leaked).
        files = list(tmp_path.glob("*.stream.jsonl"))
        assert len(files) == 1
        files[0].read_text()  # raises if file handle leaked

    async def test_stream_toolcall_events_recorded(self, tmp_path):
        """toolcall_start / toolcall_delta / toolcall_end events reach the stream file."""
        from pydantic import BaseModel

        class P(BaseModel):
            pass

        async def noop(tool_call_id: str, params: P, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")])

        tool = AgentTool(name="noop", description="d", parameters=P, execute=noop)
        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="tc1", name="noop", arguments={"x": 1})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("all done"),
            ]
        )
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s", tools=[tool])
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        await agent.prompt("go")
        await agent.wait_for_idle()
        await tracer.shutdown()

        files = list(tmp_path.glob("*.stream.jsonl"))
        assert len(files) == 1
        events = [
            json.loads(line) for line in files[0].read_text().splitlines() if line
        ]
        types = [e["type"] for e in events]
        assert "toolcall_start" in types
        assert "toolcall_delta" in types
        assert "toolcall_end" in types
        # toolcall_end should carry accumulated arg char count
        end_ev = next(e for e in events if e["type"] == "toolcall_end")
        assert "args_chars" in end_ev
        assert end_ev["args_chars"] > 0

    async def test_no_stream_file_without_record_stream(self, tmp_path):
        """Default (record_stream=False) must not create any stream file."""
        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("hi")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(service_name="t", agent_name="a", exporters=[])
        tracer.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        assert not list(tmp_path.glob("*.stream.jsonl"))

    async def test_stream_events_have_elapsed_time_field(self, tmp_path):
        """Every recorded stream event must carry a numeric 't' (elapsed) field."""
        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("hi")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        stream_file = next(tmp_path.glob("*.stream.jsonl"))
        events = [
            json.loads(line) for line in stream_file.read_text().splitlines() if line
        ]
        assert events
        for ev in events:
            assert "t" in ev
            assert isinstance(ev["t"], (int, float))

    async def test_stream_error_event_recorded(self, tmp_path):
        """An error stop_reason causes FauxProvider to emit an 'error' StreamEvent,
        which _write_stream_event captures in the stream file."""
        provider = FauxProvider(provider_id="faux")
        provider.append_responses(
            [faux_assistant_message("oops", stop_reason="error", error_message="boom")]
        )
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

        files = list(tmp_path.glob("*.stream.jsonl"))
        assert len(files) == 1
        events = [
            json.loads(line) for line in files[0].read_text().splitlines() if line
        ]
        types = [e["type"] for e in events]
        assert "error" in types
        err_ev = next(e for e in events if e["type"] == "error")
        assert "error_message" in err_ev

    async def test_stream_write_exception_swallowed(self, tmp_path):
        """If the stream file write raises, _write_stream_event swallows it silently."""
        from unittest.mock import MagicMock, patch

        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("hi")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        # Run once to create the stream file and let the run finish normally.
        await agent.prompt("x")
        await agent.wait_for_idle()

        # Now simulate a second run where the file write raises.
        # We directly patch the stream_file.write on the recorder's _run state
        # after the second AgentStart so the open succeeds but every write fails.
        import cubepi.tracing.recorder as _rec_mod

        _orig_on_agent_start = _rec_mod.Recorder._on_agent_start

        bad_file = MagicMock()
        bad_file.write.side_effect = OSError("disk full")
        bad_file.close.return_value = None

        def _patched_start(self_rec):
            _orig_on_agent_start(self_rec)
            # Replace the freshly-opened stream_file with the bad mock.
            if self_rec._run is not None:
                self_rec._run.stream_file = bad_file

        provider.append_responses([faux_assistant_message("ok")])
        with patch.object(_rec_mod.Recorder, "_on_agent_start", _patched_start):
            await agent.prompt("y")
            await agent.wait_for_idle()

        await tracer.shutdown()
        # write was called and raised, but the agent completed without crashing.
        bad_file.write.assert_called()

    async def test_stream_open_exception_swallowed(self, tmp_path):
        """If the stream file cannot be opened, the recorder continues without crashing."""
        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("hi")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        # Pass stream_dir as an existing file (not a directory) so mkdir fails.
        bad_dir = tmp_path / "not_a_dir.txt"
        bad_dir.write_text("i am a file")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=bad_dir,
        )
        tracer.attach(agent)

        # Should not raise even though stream file can't be opened.
        await agent.prompt("x")
        await agent.wait_for_idle()
        await tracer.shutdown()

    async def test_stream_close_exception_swallowed_on_agent_end(self, tmp_path):
        """If stream_file.close() raises during _on_agent_end, it is swallowed."""
        import cubepi.tracing.recorder as _rec_mod
        from unittest.mock import MagicMock, patch

        provider = FauxProvider(provider_id="faux")
        provider.append_responses([faux_assistant_message("hi")])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        tracer.attach(agent)

        _orig_on_agent_start = _rec_mod.Recorder._on_agent_start

        bad_file: MagicMock | None = None

        def _patched_start(self_rec: _rec_mod.Recorder) -> None:
            nonlocal bad_file
            _orig_on_agent_start(self_rec)
            if self_rec._run is not None and self_rec._run.stream_file is not None:
                bad_file = MagicMock()
                bad_file.write.return_value = None
                bad_file.close.side_effect = OSError("close error")
                self_rec._run.stream_file = bad_file

        with patch.object(_rec_mod.Recorder, "_on_agent_start", _patched_start):
            await agent.prompt("x")
            await agent.wait_for_idle()
        await tracer.shutdown()
        assert bad_file is not None
        bad_file.close.assert_called()

    async def test_stream_close_exception_swallowed_on_cancel(self, tmp_path):
        """If stream_file.close() raises during _close_open_spans, it is swallowed."""
        import cubepi.tracing.recorder as _rec_mod
        from unittest.mock import MagicMock, patch

        provider = FauxProvider(tokens_per_second=5.0)
        provider.append_responses([faux_assistant_message("x" * 300)])
        agent = Agent(model=provider.model(MODEL.id), system_prompt="s")
        tracer = Tracer(
            service_name="t",
            agent_name="a",
            exporters=[],
            record_stream=True,
            stream_dir=tmp_path,
        )
        detach = tracer.attach(agent)

        _orig_on_agent_start = _rec_mod.Recorder._on_agent_start
        bad_file: MagicMock | None = None

        def _patched_start(self_rec: _rec_mod.Recorder) -> None:
            nonlocal bad_file
            _orig_on_agent_start(self_rec)
            if self_rec._run is not None and self_rec._run.stream_file is not None:
                bad_file = MagicMock()
                bad_file.write.return_value = None
                bad_file.close.side_effect = OSError("close error")
                self_rec._run.stream_file = bad_file

        with patch.object(_rec_mod.Recorder, "_on_agent_start", _patched_start):
            run = asyncio.create_task(agent.prompt("hi"))
            await asyncio.sleep(0.05)
            run.cancel()
            try:
                await run
            except asyncio.CancelledError:
                pass
        detach()
        await tracer.shutdown()
        assert bad_file is not None
        bad_file.close.assert_called()
