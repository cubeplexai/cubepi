"""Tests for BoundModel.generate_structured()."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cubepi.providers.base import (
    StructuredOutputError,
    TextContent,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str


async def test_generate_structured_happy_path() -> None:
    """FauxProvider returns a tool_call with valid args → returns validated MovieReview instance."""
    provider = FauxProvider()
    model = provider.model("faux-1")
    provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"title": "Inception", "rating": 9, "summary": "Mind-bending heist film."},
                ),
                stop_reason="tool_use",
            )
        ]
    )

    messages = [UserMessage(content=[TextContent(text="Review Inception")])]
    result = await model.generate_structured(MovieReview, messages)

    assert isinstance(result, MovieReview)
    assert result.title == "Inception"
    assert result.rating == 9
    assert result.summary == "Mind-bending heist film."


async def test_generate_structured_no_tool_call_raises() -> None:
    """FauxProvider returns plain text → raises StructuredOutputError matching 'no tool call'."""
    provider = FauxProvider()
    model = provider.model("faux-1")
    provider.set_responses(
        [
            faux_assistant_message("Here is my review: it was great.", stop_reason="stop")
        ]
    )

    messages = [UserMessage(content=[TextContent(text="Review a movie")])]
    with pytest.raises(StructuredOutputError, match="no tool call"):
        await model.generate_structured(MovieReview, messages)


async def test_generate_structured_validation_error_raises() -> None:
    """FauxProvider returns invalid data twice → raises StructuredOutputError matching 'validation'."""
    provider = FauxProvider()
    model = provider.model("faux-1")
    # Two responses with invalid data (rating is not a number)
    invalid_response = faux_assistant_message(
        faux_tool_call(
            "structured_output",
            {"title": "Inception", "rating": "not-a-number", "summary": "Great film."},
        ),
        stop_reason="tool_use",
    )
    provider.set_responses([invalid_response, invalid_response])

    messages = [UserMessage(content=[TextContent(text="Review a movie")])]
    with pytest.raises(StructuredOutputError, match="validation"):
        await model.generate_structured(MovieReview, messages, max_retries=1)


async def test_generate_structured_custom_tool_name() -> None:
    """Uses tool_name='my_output', FauxProvider returns tool_call with name 'my_output' → succeeds."""
    provider = FauxProvider()
    model = provider.model("faux-1")
    provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "my_output",
                    {"title": "The Matrix", "rating": 10, "summary": "Reality-bending classic."},
                ),
                stop_reason="tool_use",
            )
        ]
    )

    messages = [UserMessage(content=[TextContent(text="Review The Matrix")])]
    result = await model.generate_structured(MovieReview, messages, tool_name="my_output")

    assert isinstance(result, MovieReview)
    assert result.title == "The Matrix"
    assert result.rating == 10


async def test_generate_structured_retry_succeeds() -> None:
    """First response has invalid data, second has valid data. With max_retries=1, succeeds on retry."""
    provider = FauxProvider()
    model = provider.model("faux-1")
    invalid_response = faux_assistant_message(
        faux_tool_call(
            "structured_output",
            {"title": "Dune", "rating": "invalid", "summary": "Epic sci-fi."},
        ),
        stop_reason="tool_use",
    )
    valid_response = faux_assistant_message(
        faux_tool_call(
            "structured_output",
            {"title": "Dune", "rating": 8, "summary": "Epic sci-fi."},
        ),
        stop_reason="tool_use",
    )
    provider.set_responses([invalid_response, valid_response])

    messages = [UserMessage(content=[TextContent(text="Review Dune")])]
    result = await model.generate_structured(MovieReview, messages, max_retries=1)

    assert isinstance(result, MovieReview)
    assert result.title == "Dune"
    assert result.rating == 8
    assert provider.call_count == 2
