import time

from collections.abc import Generator

import pytest
import requests


def is_ollama_ready(base_url: str = "http://localhost:11434", timeout: float = 1.0) -> bool:
    """Check if Ollama server is responsive."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200  # noqa: TRY300, PLR2004
    except requests.RequestException:
        return False


def check_model_available(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if specific model is available."""
    try:
        response = requests.get(f"{base_url}/api/tags")  # noqa: S113
        models = [model["name"] for model in response.json().get("models", [])]
        return model_name in models  # noqa: TRY300
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def ensure_ollama() -> Generator[None, None, None]:
    """Ensure Ollama is running and the required model is available."""
    # Check if Ollama is running
    max_retries = 5
    retry_delay = 2

    print("\nChecking Ollama availability...")
    for attempt in range(max_retries):
        if is_ollama_ready():
            break
        if attempt < max_retries - 1:
            print(f"Ollama not ready, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    else:
        pytest.skip("Ollama server is not available")

    # Check if required model is available
    model_name = "llama3:latest"
    if not check_model_available(model_name):
        pytest.skip(f"{model_name} model is not available. Please run 'ollama pull {model_name}'")

    yield


@pytest.fixture(scope="session")
def model_name() -> str:
    """Provide the model name for tests."""
    return "llama3:latest"


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    """Provide the base URL for Ollama service."""
    return "http://localhost:11434"


# Add mock fixtures for unit tests
@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "llama3:latest",
        "response": '{"messages": [{"role": "user", "content": "test"}]}',
        "total_duration": 1000000,
        "prompt_eval_count": 10,
        "eval_count": 20,
    }


@pytest.fixture
def mock_ollama_client(mock_ollama_response):
    """Mock OllamaClient."""
    from unittest.mock import Mock

    client = Mock()
    client.generate_completion.return_value = Mock(
        content=mock_ollama_response["response"],
        total_duration=mock_ollama_response["total_duration"],
        prompt_eval_count=mock_ollama_response["prompt_eval_count"],
        eval_count=mock_ollama_response["eval_count"],
    )
    return client
