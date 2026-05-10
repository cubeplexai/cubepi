from cubepi.providers import (
    get_anthropic_provider,
    get_openai_provider,
    get_openai_responses_provider,
)


class TestLazyProviderFactories:
    def test_get_anthropic_provider(self):
        cls = get_anthropic_provider()
        assert cls.__name__ == "AnthropicProvider"

    def test_get_openai_provider(self):
        cls = get_openai_provider()
        assert cls.__name__ == "OpenAIProvider"

    def test_get_openai_responses_provider(self):
        cls = get_openai_responses_provider()
        assert cls.__name__ == "OpenAIResponsesProvider"
