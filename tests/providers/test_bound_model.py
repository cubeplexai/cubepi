from cubepi.providers.base import AssistantMessage, BaseProvider, BoundModel, Model
from cubepi.providers.base import TextContent
from cubepi.providers.faux import FauxProvider


def test_base_provider_model_binds_runtime_provider_and_model_spec() -> None:
    provider = BaseProvider(provider_id="catalog-anthropic")

    bound = provider.model(
        "claude-sonnet-4-5-20250929",
        api="anthropic-messages",
        reasoning=True,
        context_window=200_000,
        max_tokens=8192,
        temperature=0.2,
    )

    assert isinstance(bound, BoundModel)
    assert bound.provider is provider
    assert bound.spec == Model(
        id="claude-sonnet-4-5-20250929",
        provider_id="catalog-anthropic",
        api="anthropic-messages",
        reasoning=True,
        context_window=200_000,
        max_tokens=8192,
        temperature=0.2,
    )


def test_base_provider_model_leaves_provider_id_empty_by_default() -> None:
    provider = BaseProvider()

    bound = provider.model("local-model")

    assert bound.provider is provider
    assert bound.spec.provider_id == ""
    assert bound.spec.id == "local-model"


def test_concrete_provider_accepts_provider_id() -> None:
    provider = FauxProvider(provider_id="faux-catalog")

    bound = provider.model("faux-1")

    assert bound.provider is provider
    assert bound.spec.provider_id == "faux-catalog"
    assert bound.spec.id == "faux-1"


def test_assistant_message_uses_model_provider_id_metadata() -> None:
    model = Model(id="faux-1", provider_id="faux-provider")

    message = AssistantMessage(
        content=[TextContent(text="ok")],
        provider_id=model.provider_id,
        model_id=model.id,
    )

    assert message.provider_id == "faux-provider"
    assert message.model_id == "faux-1"
