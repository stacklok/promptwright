def test_dataset_initialization():
    """Test Dataset class initialization."""
    from promptwright import Dataset

    dataset = Dataset()
    assert len(dataset) == 0
    assert dataset.samples == []


def test_dataset_validation():
    """Test sample validation."""
    from promptwright import Dataset

    valid_sample = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
    }

    invalid_sample = {"messages": [{"role": "invalid", "content": "test"}]}

    assert Dataset.validate_sample(valid_sample) is True
    assert Dataset.validate_sample(invalid_sample) is False


def test_dataset_add_samples():
    """Test adding samples to dataset."""
    from promptwright import Dataset

    dataset = Dataset()

    samples = [
        {
            "messages": [
                {"role": "user", "content": "test1"},
                {"role": "assistant", "content": "response1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "test2"},
                {"role": "assistant", "content": "response2"},
            ]
        },
    ]

    dataset.add_samples(samples)
    assert len(dataset) == 2  # noqa: PLR2004
    assert dataset[0] == samples[0]


def test_dataset_filter_by_role():
    """Test filtering samples by role."""
    from promptwright import Dataset

    dataset = Dataset()

    samples = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "test1"},
                {"role": "assistant", "content": "response1"},
            ]
        }
    ]

    dataset.add_samples(samples)
    user_messages = dataset.filter_by_role("user")
    assert len(user_messages) == 1
    assert user_messages[0]["messages"][0]["content"] == "test1"
