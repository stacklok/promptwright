import pytest

from promptwright import LocalDataEngine, LocalEngineArguments  # Updated import


def test_engine_initialization():
    """Test LocalDataEngine initialization."""
    args = LocalEngineArguments(
        instructions="Test instructions",
        system_prompt="Test system prompt",
        model_name="llama3:latest",
    )

    engine = LocalDataEngine(args)
    assert engine.args == args
    assert len(engine.dataset) == 0
    assert engine.failed_samples == []


@pytest.mark.usefixtures("mock_ollama_client")
def test_engine_create_data(mock_ollama_client):
    """Test create_data method."""
    args = LocalEngineArguments(
        instructions="Test instructions",
        system_prompt="Test system prompt",
        model_name="llama3:latest",
    )

    engine = LocalDataEngine(args)
    engine.llm_client = mock_ollama_client

    dataset = engine.create_data(num_steps=1, batch_size=1)
    assert len(dataset) == 1


def test_engine_validation():
    """Test sample validation in engine."""
    args = LocalEngineArguments(
        instructions="Test instructions",
        system_prompt="Test system prompt",
        model_name="llama3:latest",
    )

    engine = LocalDataEngine(args)

    valid_sample = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
    }

    invalid_sample = {"messages": [{"role": "invalid", "content": ""}]}

    assert engine._validate_sample(valid_sample) is True
    assert engine._validate_sample(invalid_sample) is False
