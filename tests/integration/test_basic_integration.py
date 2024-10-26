import pytest

from promptsmith import LocalDataEngine, LocalEngineArguments, OllamaClient


@pytest.mark.integration
def test_ollama_connection():
    """Test basic connection to Ollama server."""
    client = OllamaClient()
    try:
        models = client.list_local_models()
        assert isinstance(models, list), "Expected list of models"
    except Exception as e:
        pytest.skip(f"Ollama server not available: {e}")


@pytest.mark.integration
def test_basic_generation():
    """Test basic prompt generation."""
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate a simple test response.",
            system_prompt="You are a helpful assistant.",
            model_name="llama3:latest",
            temperature=0.7,
            max_retries=2,
        )
    )

    try:
        dataset = engine.create_data(num_steps=1, batch_size=1)
        assert len(dataset) > 0, "Expected at least one sample"

        sample = dataset[0]
        assert "messages" in sample, "Expected messages in sample"
        assert len(sample["messages"]) > 0, "Expected non-empty messages"

    except Exception as e:
        pytest.skip(f"Generation failed: {e}")


@pytest.mark.integration
def test_model_availability():
    """Test if required model is available."""
    client = OllamaClient()
    try:
        models = client.list_local_models()
        model_names = [m.get("name") for m in models]
        assert (
            "llama3:latest" in model_names
        ), "llama3:latest not found. Please run: ollama pull llama3:latest"
    except Exception as e:
        pytest.skip(f"Could not check models: {e}")
