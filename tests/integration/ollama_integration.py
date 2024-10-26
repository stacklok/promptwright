import pytest

from promptsmith import LocalDataEngine, LocalEngineArguments, OllamaClient


@pytest.mark.integration
def test_ollama_basic_completion(ensure_ollama):  # noqa: ARG001
    """Test basic completion with Ollama."""
    client = OllamaClient()

    response = client.generate_completion(
        prompt="What is 2+2?", model="llama3:latest", temperature=0.7
    )

    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.integration
def test_data_engine_generation(ensure_ollama):  # noqa: ARG001
    """Test full data generation workflow."""
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate simple math questions and answers.",
            system_prompt="You are a math tutor. Provide clear, concise answers.",
            model_name="llama3:latest",
            temperature=0.7,
            max_retries=2,
        )
    )

    dataset = engine.create_data(num_steps=2, batch_size=1)

    assert len(dataset) > 0
    for sample in dataset:
        assert "messages" in sample
        assert len(sample["messages"]) > 0
        assert all(msg["role"] in ["user", "assistant", "system"] for msg in sample["messages"])
