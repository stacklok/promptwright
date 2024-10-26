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
    model_name = "llama3:latest"  # You can make this configurable if needed
    print(f"\nChecking if {model_name} is available...")
    if not check_model_available(model_name):
        pytest.skip(f"{model_name} model is not available. Please run 'ollama pull {model_name}'")

    print(f"\n{model_name} is available and ready for testing.")
    yield

    # Cleanup code (if needed) would go here
    print("\nCleaning up after tests...")


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    """Provide the base URL for Ollama service."""
    return "http://localhost:11434"


@pytest.fixture(scope="session")
def model_name() -> str:
    """Provide the model name for tests."""
    return "llama3:latest"
