import os
import tempfile

import pytest

from promptsmith import Dataset, LocalDataEngine, LocalEngineArguments

pytestmark = pytest.mark.integration


def test_dataset_persistence(ensure_ollama, model_name):  # noqa: ARG001
    """Test dataset saving and loading with integration."""
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate a simple test example.",
            system_prompt="You are a helpful assistant.",
            model_name=model_name,
            temperature=0.7,
        )
    )

    try:
        # Generate minimal dataset
        dataset = engine.create_data(num_steps=1, batch_size=1)
        assert len(dataset) > 0, "Failed to generate dataset"

        # Test save/load cycle
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            dataset.save(tmp.name)
            loaded = Dataset.from_jsonl(tmp.name)
            assert len(loaded) == len(dataset), "Dataset size changed after save/load"
            os.unlink(tmp.name)

    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")


def test_end_to_end_workflow(ensure_ollama, model_name, ollama_base_url):  # noqa: ARG001
    """Test complete workflow."""
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate Python coding examples.",
            system_prompt="You are a Python expert.",
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            temperature=0.7,
        )
    )

    try:
        dataset = engine.create_data(num_steps=1, batch_size=1)
        assert len(dataset) > 0, "Dataset should not be empty"

        sample = dataset[0]
        assert "messages" in sample, "Sample should contain messages"
        assert len(sample["messages"]) > 0, "Sample should have messages"

    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")
