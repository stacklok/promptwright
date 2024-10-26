import json
import random
import time

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from tqdm import tqdm

from .dataset import Dataset
from .ollama_client import OllamaClient

# Handle circular import for type hints
if TYPE_CHECKING:
    from .topic_tree import LocalTopicTree


@dataclass
class LocalEngineArguments:
    instructions: str
    system_prompt: str
    model_name: str
    prompt_template: str | None = None
    example_data: Dataset | None = None
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_retries: int = 3
    default_batch_size: int = 5
    default_num_examples: int = 3
    request_timeout: int = 30  # Added timeout parameter


class LocalDataEngine:
    DEFAULT_PROMPT_TEMPLATE = """Generate a JSON object in this exact format:
    {
        "messages": [
            {
                "role": "user",
                "content": "<question>"
            },
            {
                "role": "assistant",
                "content": "<answer>"
            }
        ]
    }"""

    def __init__(self, args: LocalEngineArguments):
        if not args.model_name:
            raise ValueError("model_name unspecified in LocalEngineArguments")  # noqa: TRY003

        self.args = args
        self.dataset = Dataset()
        self.llm_client = OllamaClient(base_url=args.ollama_base_url)
        self.failed_samples = []  # Track failed attempts

    def create_data(  # noqa: PLR0912
        self,  # noqa: PLR0912
        num_steps: int | None = None,
        batch_size: int | None = None,
        num_example_demonstrations: int | None = None,
        topic_tree: Optional["LocalTopicTree"] = None,
    ) -> Dataset:
        """Generate training data with improved reliability."""
        batch_size = min(batch_size or self.args.default_batch_size, 5)

        if num_steps is None and topic_tree is None:
            raise ValueError("Must specify either num_steps or provide a topic_tree")  # noqa: TRY003

        total_samples = num_steps * batch_size
        success_count = 0
        consecutive_failures = 0

        print("\nStarting generation:")
        print(f"Target: {total_samples} samples")
        print(f"Model: {self.args.model_name}")
        print(f"Batch size: {batch_size}")

        start_time = time.time()
        last_save_time = start_time

        try:
            with tqdm(total=total_samples, desc="Generating samples") as pbar:
                for step in range(num_steps):
                    for batch_item in range(batch_size):
                        sample_success = False
                        retries = 0

                        while not sample_success and retries < self.args.max_retries:
                            try:
                                # Generate prompt
                                prompt = self.build_prompt(
                                    num_example_demonstrations=(num_example_demonstrations or 0),
                                    subtopics_list=None,
                                )

                                # Get model response with timeout
                                response = self.llm_client.generate_completion(
                                    prompt=prompt,
                                    model=self.args.model_name,
                                    system_prompt=self.args.system_prompt,
                                    temperature=self.args.temperature,
                                )

                                # Parse and validate
                                sample = json.loads(response.content)
                                if self._validate_sample(sample):
                                    self.dataset.add_samples([sample])
                                    success_count += 1
                                    consecutive_failures = 0
                                    sample_success = True
                                    pbar.update(1)
                                else:
                                    retries += 1

                            except Exception as e:
                                retries += 1
                                consecutive_failures += 1
                                self.failed_samples.append(
                                    {"step": step, "batch_item": batch_item, "error": str(e)}
                                )

                                if consecutive_failures >= 5:  # noqa: PLR2004
                                    print(
                                        f"\nToo many consecutive failures ({consecutive_failures}). Saving progress..."
                                    )
                                    self.save_dataset(
                                        f"emergency_save_{success_count}_samples.jsonl"
                                    )
                                    print("Consider:")
                                    print("1. Checking Ollama status")
                                    print("2. Restarting Ollama")
                                    print("3. Using a different model")
                                    return self.dataset

                    # Save progress every 5 minutes or 10 successful samples
                    current_time = time.time()
                    if (current_time - last_save_time) > 300 or success_count % 10 == 0:  # noqa: PLR2004
                        last_save_time = current_time

                        # Show progress statistics
                        elapsed = current_time - start_time
                        rate = success_count / elapsed
                        eta = (total_samples - success_count) / rate if rate > 0 else 0

                        print("\nProgress update:")
                        print(f"Samples generated: {success_count}/{total_samples}")
                        print(f"Generation rate: {rate:.2f} samples/second")
                        print(f"Estimated time remaining: {eta/60:.1f} minutes")

        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
            self.save_dataset("interrupted_dataset.jsonl")

        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            self.save_dataset("error_dataset.jsonl")
            raise

        finally:
            # Save failure log if there were any failures
            if self.failed_samples:
                with open("generation_failures.json", "w") as f:
                    json.dump(self.failed_samples, f, indent=2)

        total_duration = time.time() - start_time
        print("\nGeneration complete:")
        print(f"Total samples: {success_count}/{total_samples}")
        print(f"Success rate: {(success_count/total_samples)*100:.1f}%")
        print(f"Total time: {total_duration/60:.1f} minutes")
        print(f"Average speed: {success_count/total_duration:.2f} samples/second")

        return self.dataset

    def build_prompt(
        self, num_example_demonstrations: int, subtopics_list: list[str] = None
    ) -> str:
        """Build a minimal, focused prompt."""
        components = []
        base_prompt = (
            self.args.prompt_template if self.args.prompt_template else self.DEFAULT_PROMPT_TEMPLATE
        )
        components.append(base_prompt)

        if self.args.instructions:
            components.append(f"\nRequirements: {self.args.instructions}")

        if subtopics_list:
            components.append(f"\nTopic: {' -> '.join(subtopics_list)}")

        # Only add examples if specifically requested
        if self.args.example_data and num_example_demonstrations > 0:
            components.append("\nExamples:")
            examples = random.sample(
                self.args.example_data.samples,
                min(num_example_demonstrations, len(self.args.example_data.samples)),
            )
            for ex in examples:
                components.append(json.dumps(ex))

        return "\n".join(components)

    def _validate_sample(self, sample: dict) -> bool:  # noqa: PLR0911
        """Validate sample format."""
        try:
            if "messages" not in sample:
                return False

            for msg in sample["messages"]:
                if not all(key in msg for key in ["role", "content"]):
                    return False
                if msg["role"] not in ["user", "assistant", "system"]:
                    return False
                if not isinstance(msg["content"], str):
                    return False
                if not msg["content"].strip():
                    return False

            return True  # noqa: TRY300
        except Exception:
            return False

    def save_dataset(self, save_path: str):
        """Save the dataset to a file."""
        self.dataset.save(save_path)
