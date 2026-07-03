from cubepi.providers.base import Model
from cubepi.providers.models import models_are_equal


def test_models_are_equal_by_provider_and_model_id():
    assert models_are_equal(
        Model(id="gpt-4o", provider_id="openai"),
        Model(id="gpt-4o", provider_id="openai"),
    )
    assert not models_are_equal(
        Model(id="gpt-4o", provider_id="openai"),
        Model(id="gpt-4o", provider_id="azure"),
    )
    assert not models_are_equal(Model(id="gpt-4o"), None)
    assert models_are_equal(None, None)
