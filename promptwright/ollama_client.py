import json

from dataclasses import dataclass

import requests


@dataclass
class LLMResponse:
    content: str
    total_duration: int
    prompt_eval_count: int
    eval_count: int


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate_completion(
        self,
        prompt: str,
        model: str = "llama2",
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate completion using the Ollama API."""

        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Disable streaming for simpler handling
            "format": "json",  # Request JSON format
            "options": {
                "temperature": temperature,
                "num_predict": 1000,
                "stop": ["\n\n", "```"],  # Stop tokens to prevent extra content
            },
        }

        if system_prompt:
            data["system"] = system_prompt

        url = f"{self.base_url}/api/generate"

        try:
            print("\nSending request to Ollama...")
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()

            # Debug output
            print(f"Raw response: {result.get('response', '')[:500]}...")

            if not result.get("response"):
                raise ValueError("Empty response from model")  # noqa: TRY003

            # Try to parse the response as JSON
            try:
                json_content = json.loads(result["response"])
                # If successful, convert back to string with proper formatting
                content = json.dumps(json_content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the response
                content = self._extract_json(result["response"])

            return LLMResponse(
                content=content,
                total_duration=result.get("total_duration", 0),
                prompt_eval_count=result.get("prompt_eval_count", 0),
                eval_count=result.get("eval_count", 0),
            )

        except requests.exceptions.Timeout:
            raise TimeoutError("Request to Ollama timed out")  # noqa: B904, TRY003
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")  # noqa: B904, TRY002, TRY003

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text."""
        try:
            # Find the first opening brace
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in response")  # noqa: TRY301, TRY003

            # Keep track of braces
            count = 0
            for i, char in enumerate(text[start:]):
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                    if count == 0:
                        # Found complete JSON object
                        json_str = text[start : start + i + 1]
                        # Validate it's proper JSON
                        json.loads(json_str)
                        return json_str

            raise ValueError("No complete JSON object found")  # noqa: TRY301, TRY003

        except Exception as e:
            raise ValueError(f"Failed to extract JSON: {str(e)}")  # noqa: B904, TRY002, TRY003

    def list_local_models(self) -> list[dict]:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            raise Exception(f"Failed to list models: {str(e)}")  # noqa: B904, TRY002, TRY003
