import os
import tempfile

import pytest

from promptsmith import Dataset


@pytest.mark.integration
def test_dataset_save_load():
    """Test dataset saving and loading with real files."""
    # Create a test dataset
    dataset = Dataset()
    test_samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4"},
            ]
        }
    ]
    dataset.add_samples(test_samples)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        dataset.save(tmp.name)

        # Load the dataset back
        loaded_dataset = Dataset.from_jsonl(tmp.name)

        # Verify content
        assert len(loaded_dataset) == len(dataset)
        assert loaded_dataset.samples == dataset.samples

    # Clean up
    os.unlink(tmp.name)
