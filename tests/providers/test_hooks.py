"""Tests for on_payload and on_response Provider hooks."""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from cubepi.providers.base import (
    Model,
    ProviderResponse,
    StreamOptions,
    invoke_on_payload,
    invoke_on_response,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _model(**overrides: Any) -> Model:
    defaults = {"id": "test-model", "provider": "test"}
    defaults.update(overrides)
    return Model(**defaults)


# ---------------------------------------------------------------------------
# ProviderResponse dataclass
# ---------------------------------------------------------------------------


class TestProviderResponse:
    def test_basic(self):
        r = ProviderResponse(status=200, headers={"x-request-id": "abc"})
        assert r.status == 200
        assert r.headers == {"x-request-id": "abc"}

    def test_default_headers(self):
        r = ProviderResponse(status=201)
        assert r.headers == {}


# ---------------------------------------------------------------------------
# invoke_on_payload helper
# ---------------------------------------------------------------------------


class TestInvokeOnPayload:
    async def test_none_callback_returns_original(self):
        payload = {"model": "test"}
        result = await invoke_on_payload(None, payload, _model())
        assert result is payload

    async def test_sync_callback_returning_dict_replaces(self):
        replacement = {"model": "replaced"}
        result = await invoke_on_payload(
            lambda p, m: replacement, {"model": "original"}, _model()
        )
        assert result is replacement

    async def test_sync_callback_returning_none_keeps_original(self):
        original = {"model": "original"}
        result = await invoke_on_payload(lambda p, m: None, original, _model())
        assert result is original

    async def test_async_callback_returning_dict_replaces(self):
        replacement = {"model": "async-replaced"}

        async def cb(p: dict, m: Model) -> dict:
            return replacement

        result = await invoke_on_payload(cb, {"model": "original"}, _model())
        assert result is replacement

    async def test_async_callback_returning_none_keeps_original(self):
        original = {"model": "original"}

        async def cb(p: dict, m: Model) -> None:
            return None

        result = await invoke_on_payload(cb, original, _model())
        assert result is original

    async def test_callback_receives_correct_args(self):
        received: list[tuple[dict, Model]] = []

        def cb(p: dict, m: Model) -> None:
            received.append((p, m))
            return None

        payload = {"model": "test-model"}
        model = _model()
        await invoke_on_payload(cb, payload, model)
        assert len(received) == 1
        assert received[0][0] is payload
        assert received[0][1] is model


# ---------------------------------------------------------------------------
# invoke_on_response helper
# ---------------------------------------------------------------------------


class TestInvokeOnResponse:
    async def test_none_callback_is_noop(self):
        # Should not raise
        await invoke_on_response(None, ProviderResponse(status=200), _model())

    async def test_sync_callback_called(self):
        received: list[tuple[ProviderResponse, Model]] = []

        def cb(r: ProviderResponse, m: Model) -> None:
            received.append((r, m))

        resp = ProviderResponse(status=200, headers={"h": "v"})
        model = _model()
        await invoke_on_response(cb, resp, model)
        assert len(received) == 1
        assert received[0][0] is resp
        assert received[0][1] is model

    async def test_async_callback_called(self):
        received: list[ProviderResponse] = []

        async def cb(r: ProviderResponse, m: Model) -> None:
            received.append(r)

        resp = ProviderResponse(status=200)
        await invoke_on_response(cb, resp, _model())
        assert len(received) == 1
        assert received[0] is resp


# ---------------------------------------------------------------------------
# AnthropicProvider hook integration
# ---------------------------------------------------------------------------


class TestAnthropicProviderHooks:
    async def test_on_payload_called_with_kwargs(self):
        """on_payload receives the payload dict and model before the API call."""
        from cubepi.providers.anthropic import AnthropicProvider
        from cubepi.providers.base import TextContent, UserMessage

        captured_payloads: list[tuple[dict, Model]] = []

        def on_payload(payload: dict, model: Model) -> None:
            captured_payloads.append((payload.copy(), model))
            return None  # don't replace

        # Mock the anthropic client
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = MagicMock(return_value=iter([]))
        mock_stream.response = MagicMock(status_code=200, headers={})

        final_msg = MagicMock()
        final_msg.content = [MagicMock(type="text", text="hello")]
        final_msg.stop_reason = "end_turn"
        final_msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_stream.get_final_message.return_value = final_msg

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            provider = AnthropicProvider(api_key="test")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            # Drain stream to let _produce run
            async for _ in ms:
                pass

        assert len(captured_payloads) == 1
        payload, recv_model = captured_payloads[0]
        assert payload["model"] == "test-model"
        assert "messages" in payload
        assert recv_model is model

    async def test_on_payload_replaces_kwargs(self):
        """When on_payload returns a dict, that dict replaces the payload."""
        from cubepi.providers.anthropic import AnthropicProvider
        from cubepi.providers.base import TextContent, UserMessage

        def on_payload(payload: dict, model: Model) -> dict:
            payload["model"] = "replaced-model"
            return payload

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = MagicMock(return_value=iter([]))
        mock_stream.response = MagicMock(status_code=200, headers={})

        final_msg = MagicMock()
        final_msg.content = [MagicMock(type="text", text="ok")]
        final_msg.stop_reason = "end_turn"
        final_msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_stream.get_final_message.return_value = final_msg

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.stream.return_value = mock_stream

            provider = AnthropicProvider(api_key="test")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            async for _ in ms:
                pass

        # Verify the replaced model was sent to the API
        call_kwargs = mock_client.messages.stream.call_args
        assert call_kwargs is not None
        assert call_kwargs[1]["model"] == "replaced-model"


# ---------------------------------------------------------------------------
# OpenAIProvider hook integration
# ---------------------------------------------------------------------------


class TestOpenAIProviderHooks:
    async def test_on_payload_called_with_kwargs(self):
        """on_payload receives the payload dict and model before the API call."""
        from cubepi.providers.openai import OpenAIProvider
        from cubepi.providers.base import TextContent, UserMessage

        captured_payloads: list[tuple[dict, Model]] = []

        def on_payload(payload: dict, model: Model) -> None:
            captured_payloads.append((payload.copy(), model))
            return None

        # Build a mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock(content="hello", tool_calls=None)
        mock_chunk.choices[0].finish_reason = None

        mock_final = MagicMock()
        mock_final.choices = [MagicMock()]
        mock_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        mock_final.choices[0].finish_reason = "stop"

        async def _aiter():
            yield mock_chunk
            yield mock_final

        mock_response = _aiter()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = OpenAIProvider(api_key="test")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            async for _ in ms:
                pass

        assert len(captured_payloads) == 1
        payload, recv_model = captured_payloads[0]
        assert payload["model"] == "test-model"
        assert recv_model is model

    async def test_on_payload_replaces_kwargs(self):
        """When on_payload returns a dict, that dict replaces the payload."""
        from cubepi.providers.openai import OpenAIProvider
        from cubepi.providers.base import TextContent, UserMessage

        def on_payload(payload: dict, model: Model) -> dict:
            payload["temperature"] = 0.5
            return payload

        mock_final = MagicMock()
        mock_final.choices = [MagicMock()]
        mock_final.choices[0].delta = MagicMock(content=None, tool_calls=None)
        mock_final.choices[0].finish_reason = "stop"

        async def _aiter():
            yield mock_final

        mock_response = _aiter()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            provider = OpenAIProvider(api_key="test")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            async for _ in ms:
                pass

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs is not None
        assert call_kwargs[1]["temperature"] == 0.5


# ---------------------------------------------------------------------------
# OpenAIResponsesProvider hook integration
# ---------------------------------------------------------------------------


class TestOpenAIResponsesProviderHooks:
    async def test_on_payload_called_with_kwargs(self):
        """on_payload receives the payload dict and model before the API call."""
        from cubepi.providers.openai_responses import OpenAIResponsesProvider
        from cubepi.providers.base import TextContent, UserMessage

        captured_payloads: list[tuple[dict, Model]] = []

        def on_payload(payload: dict, model: Model) -> None:
            captured_payloads.append((payload.copy(), model))
            return None

        # Build a mock streaming response that emits response.completed
        mock_completed_event = MagicMock()
        mock_completed_event.type = "response.completed"
        mock_completed_event.response = MagicMock()
        mock_completed_event.response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=None,
        )
        mock_completed_event.response.status = "completed"
        mock_completed_event.response.id = "resp_123"

        async def _aiter():
            yield mock_completed_event

        mock_response = _aiter()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            provider = OpenAIResponsesProvider(api_key="test")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            async for _ in ms:
                pass

        assert len(captured_payloads) == 1
        payload, recv_model = captured_payloads[0]
        assert payload["model"] == "test-model"
        assert "input" in payload
        assert recv_model is model

    async def test_on_payload_replaces_kwargs(self):
        """When on_payload returns a dict, that dict replaces the payload."""
        from cubepi.providers.openai_responses import OpenAIResponsesProvider
        from cubepi.providers.base import TextContent, UserMessage

        def on_payload(payload: dict, model: Model) -> dict:
            payload["temperature"] = 0.7
            return payload

        mock_completed_event = MagicMock()
        mock_completed_event.type = "response.completed"
        mock_completed_event.response = MagicMock()
        mock_completed_event.response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=None,
        )
        mock_completed_event.response.status = "completed"
        mock_completed_event.response.id = "resp_123"

        async def _aiter():
            yield mock_completed_event

        mock_response = _aiter()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.responses.create = AsyncMock(return_value=mock_response)

            provider = OpenAIResponsesProvider(api_key="test")
            provider._client = mock_client

            ms = await provider.stream(
                _model(),
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_payload=on_payload),
            )
            async for _ in ms:
                pass

        call_kwargs = mock_client.responses.create.call_args
        assert call_kwargs is not None
        assert call_kwargs[1]["temperature"] == 0.7

    async def test_on_response_called_when_http_response_available(self):
        """on_response is invoked with HTTP metadata after the API call."""
        from cubepi.providers.openai_responses import OpenAIResponsesProvider
        from cubepi.providers.base import TextContent, UserMessage

        captured_responses: list[tuple[ProviderResponse, Model]] = []

        def on_response(resp: ProviderResponse, model: Model) -> None:
            captured_responses.append((resp, model))

        mock_completed_event = MagicMock()
        mock_completed_event.type = "response.completed"
        mock_completed_event.response = MagicMock()
        mock_completed_event.response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=None,
        )
        mock_completed_event.response.status = "completed"
        mock_completed_event.response.id = "resp_123"

        # Use a MagicMock with __aiter__ so we can attach .response
        mock_stream = MagicMock()

        async def _aiter_impl():
            yield mock_completed_event

        mock_stream.__aiter__ = MagicMock(return_value=_aiter_impl().__aiter__())

        # Attach a fake HTTP response to the stream object
        mock_http = MagicMock()
        mock_http.status_code = 200
        mock_http.headers = {"x-request-id": "req_abc"}
        mock_stream.response = mock_http

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.responses.create = AsyncMock(return_value=mock_stream)

            provider = OpenAIResponsesProvider(api_key="test")
            provider._client = mock_client

            model = _model()
            ms = await provider.stream(
                model,
                [UserMessage(content=[TextContent(text="hi")])],
                options=StreamOptions(on_response=on_response),
            )
            async for _ in ms:
                pass

        assert len(captured_responses) == 1
        resp, recv_model = captured_responses[0]
        assert resp.status == 200
        assert resp.headers == {"x-request-id": "req_abc"}
        assert recv_model is model


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_provider_protocol_accepts_options(self):
        """The Provider protocol's stream method includes options."""
        from cubepi.providers.base import Provider

        sig = inspect.signature(Provider.stream)
        params = sig.parameters
        assert "options" in params
        assert params["options"].default is None
