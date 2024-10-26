from unittest.mock import Mock

import pytest
import requests

from promptwright import OllamaClient


def test_ollama_client_initialization():
    """Test OllamaClient initialization."""
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"

    client = OllamaClient("http://custom:1234")
    assert client.base_url == "http://custom:1234"


def test_generate_completion(mock_ollama_response, mocker):
    """Test generate_completion method."""
    # Setup mock
    mock_post = mocker.patch("requests.post")
    mock_post.return_value = Mock()
    mock_post.return_value.json.return_value = mock_ollama_response
    mock_post.return_value.raise_for_status = Mock()

    client = OllamaClient()
    response = client.generate_completion(prompt="Test prompt", model="llama3:latest")

    assert response.content is not None
    assert isinstance(response.content, str)
    mock_post.assert_called_once()

    # Verify the request
    call_args = mock_post.call_args
    assert call_args is not None
    args, kwargs = call_args
    assert args[0].endswith("/generate")  # Verify endpoint
    assert kwargs["json"]["prompt"] == "Test prompt"  # Verify prompt
    assert kwargs["json"]["model"] == "llama3:latest"  # Verify model


def test_generate_completion_error(mocker):
    """Test generate_completion error handling."""
    # Setup mock to raise Timeout
    _mock_post = mocker.patch("requests.post", side_effect=requests.exceptions.Timeout())

    client = OllamaClient()
    with pytest.raises(TimeoutError):
        client.generate_completion(prompt="Test prompt", model="llama3:latest")
