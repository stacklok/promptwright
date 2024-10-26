import time

import requests


def wait_for_ollama(timeout: int = 30, base_url: str = "http://localhost:11434") -> bool:
    """Wait for Ollama server to be ready.

    Args:
        timeout: Maximum time to wait in seconds
        base_url: Ollama server URL

    Returns:
        bool: True if server is ready, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/api/tags")  # noqa: S113
            if response.status_code == 200:  # noqa: PLR2004
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def ensure_model_exists(model_name: str, base_url: str = "http://localhost:11434") -> str | None:
    """Ensure a model exists in Ollama.

    Args:
        model_name: Name of the model to check
        base_url: Ollama server URL

    Returns:
        str: Error message if there's a problem, None if model exists
    """
    try:
        response = requests.get(f"{base_url}/api/tags")  # noqa: S113
        if response.status_code != 200:  # noqa: PLR2004
            return f"Failed to get models list: {response.status_code}"

        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]

        if model_name not in model_names:
            return f"Model {model_name} not found. Run: ollama pull {model_name}"

        return None  # noqa: TRY300
    except requests.RequestException as e:
        return f"Failed to check model: {str(e)}"
