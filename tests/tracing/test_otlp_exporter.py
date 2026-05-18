"""Phase 3: pin the OTLPSpanExporter re-export contract + OpenAI-specific
provider attributes.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SpanExporter

from cubepi.tracing.recorder import Recorder
from cubepi.tracing.tracer import Tracer


class _Span:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


class TestOTLPSpanExporterReExport:
    def test_otlp_exporter_is_importable(self):
        from cubepi.tracing.exporters import OTLPSpanExporter

        # Returned class must be the actual OTLP exporter.
        assert OTLPSpanExporter.__module__.startswith(
            "opentelemetry.exporter.otlp.proto.http"
        )

    def test_otlp_exporter_can_be_constructed(self):
        from cubepi.tracing.exporters import OTLPSpanExporter

        # Construct with default kwargs — should not require a live
        # collector at construction time.
        exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
        assert isinstance(exporter, SpanExporter)
        # It honors the standard SpanExporter ABC.
        exporter.shutdown()

    def test_unknown_attr_still_raises(self):
        from cubepi.tracing import exporters

        with pytest.raises(AttributeError):
            _ = exporters.NotAnExporter  # type: ignore[attr-defined]


class TestOpenAIRequestAttrs:
    def test_service_tier_recorded_on_request(self):
        # Direct unit-level: build a fake payload with service_tier
        # and feed _on_provider_request via the Recorder.
        from cubepi.providers.base import Model
        from cubepi.tracing import Tracer

        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)
        # Bootstrap a minimal run so _on_provider_request has a turn_span.
        recorder._on_agent_start()
        recorder._on_turn_start()
        payload = {
            "model": "gpt-test",
            "messages": [],
            "service_tier": "scale",
        }
        recorder._on_provider_request(payload, Model(id="gpt-test", provider="openai"))
        chat = recorder._run.chat_span  # type: ignore[union-attr]
        # chat_span is an actual SDK Span; read via .attributes
        attrs = dict(chat.attributes or {})
        assert attrs.get("openai.request.service_tier") == "scale"


class TestOpenAIResponseAttrs:
    def test_chat_completion_records_openai_provider_specifics(self):
        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)
        span = _Span()
        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-test",
            "system_fingerprint": "fp_abc123",
            "service_tier": "scale",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        recorder._record_chat_response_attrs(span, body)
        assert span.attrs["openai.api.type"] == "chat_completions"
        assert span.attrs["openai.response.system_fingerprint"] == "fp_abc123"
        assert span.attrs["openai.response.service_tier"] == "scale"

    def test_responses_api_records_api_type(self):
        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)
        span = _Span()
        body = {
            "id": "resp_1",
            "object": "response",
            "model": "o4-test",
            "status": "completed",
            "service_tier": "default",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        recorder._record_chat_response_attrs(span, body)
        assert span.attrs["openai.api.type"] == "responses"
        assert span.attrs["openai.response.service_tier"] == "default"
        # Responses API doesn't emit system_fingerprint.
        assert "openai.response.system_fingerprint" not in span.attrs

    def test_no_openai_fields_when_absent(self):
        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)
        span = _Span()
        body = {
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [{"finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        recorder._record_chat_response_attrs(span, body)
        # openai.api.type is always set for chat.completion bodies.
        assert span.attrs["openai.api.type"] == "chat_completions"
        assert "openai.response.system_fingerprint" not in span.attrs
        assert "openai.response.service_tier" not in span.attrs


class TestAnthropicShapeUnchanged:
    """Adding OpenAI-specific fields must NOT pollute Anthropic-shape
    handling — codex-style regression guard."""

    def test_anthropic_body_does_not_get_openai_attrs(self):
        tracer = Tracer(service_name="t", exporters=[])
        recorder = Recorder(tracer)
        span = _Span()
        body = {
            "id": "msg_1",
            "model": "claude-test",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        recorder._record_chat_response_attrs(span, body)
        for forbidden in (
            "openai.api.type",
            "openai.response.system_fingerprint",
            "openai.response.service_tier",
        ):
            assert forbidden not in span.attrs
