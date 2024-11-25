import json
import math
import random
import re

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import litellm

from tqdm import tqdm

from .dataset import Dataset
from .prompts import ENGINE_JSON_INSTRUCTIONS, SAMPLE_GENERATION_PROMPT
from .topic_tree import TopicTree

# Handle circular import for type hints
if TYPE_CHECKING:
    from .topic_tree import TopicTree


def validate_json_response(
    json_str: str, schema: dict[str, Any] | None = None
) -> dict | None:
    """Validate and clean JSON response from LLM."""
    try:
        json_match = re.search(r"(?s)\{.*\}", json_str)
        if not json_match:
            return None

        cleaned_json = json_match.group(0)
        cleaned_json = re.sub(r"```json\s*|\s*```", "", cleaned_json)

        parsed = json.loads(cleaned_json)

        if schema is not None:
            # Schema validation could be added here
            pass
        else:
            return parsed
    except (json.JSONDecodeError, ValueError):
        return None


@dataclass
class EngineArguments:
    instructions: str
    system_prompt: str
    model_name: str  # Required field
    prompt_template: str | None = None
    example_data: Dataset | None = None
    temperature: float = 0.2
    max_retries: int = 3
    default_batch_size: int = 5
    default_num_examples: int = 3
    request_timeout: int = 30
    sys_msg: bool = True  # Default to True for including system message


class DataEngine:
    def __init__(self, args: EngineArguments):
        if (
            not args.model_name
            or not isinstance(args.model_name, str)
            or not args.model_name.strip()
        ):
            raise ValueError(  # noqa: TRY003
                "model_name must be a non-empty string in EngineArguments"
            )  # noqa: TRY003

        self.model_name = (
            args.model_name.strip()
        )  # Store model_name as instance variable
        self.args = args
        self.dataset = Dataset()
        self.failed_samples = []
        self.failure_analysis = {
            "json_parsing_errors": [],
            "invalid_schema": [],
            "api_errors": [],
            "empty_responses": [],
            "malformed_responses": [],
            "other_errors": [],
        }
        # Store original system prompt for dataset inclusion
        self.original_system_prompt = args.system_prompt
        # Use ENGINE_JSON_INSTRUCTIONS only for generation prompt
        self.generation_system_prompt = ENGINE_JSON_INSTRUCTIONS + args.system_prompt

    def analyze_failure(self, response_content: str, error: Exception = None) -> str:
        """Analyze the failure reason for a sample."""
        if error:
            error_str = str(error)
            if "schema" in error_str.lower():
                return "invalid_schema"
            if any(
                api_err in error_str.lower()
                for api_err in ["timeout", "rate limit", "connection"]
            ):
                return "api_errors"
            return "other_errors"

        if not response_content or response_content.isspace():
            return "empty_responses"

        # Check if response seems to be attempting JSON but failing
        if any(char in response_content for char in "{}[]"):
            return "json_parsing_errors"
        return "malformed_responses"

    def summarize_failures(self) -> dict:
        """Generate a summary of all failures."""
        summary = {
            "total_failures": len(self.failed_samples),
            "failure_types": {k: len(v) for k, v in self.failure_analysis.items()},
            "failure_examples": {},
        }

        # Add example failures for each category
        for category, failures in self.failure_analysis.items():
            if failures:
                # Get up to 3 examples for each category
                examples = failures[:3]
                summary["failure_examples"][category] = [
                    (
                        str(ex)[:200] + "..."
                        if len(str(ex)) > 200  # noqa: PLR2004
                        else str(ex)  # noqa: PLR2004
                    )  # noqa: PLR2004
                    for ex in examples
                ]
        return summary

    def create_data(  # noqa: PLR0912
        self,
        num_steps: int = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_tree: TopicTree = None,
        model_name: str = None,
        sys_msg: bool = None,  # Allow overriding sys_msg from args
    ):
        if num_steps is None:
            raise ValueError("num_steps must be specified")  # noqa: TRY003

        # Use instance model_name as fallback if none provided
        self.model_name = model_name.strip() if model_name else self.model_name

        if not self.model_name:
            raise ValueError("No valid model_name provided")  # noqa: TRY003

        # Use provided sys_msg or fall back to args.sys_msg
        include_sys_msg = sys_msg if sys_msg is not None else self.args.sys_msg

        data_creation_prompt = SAMPLE_GENERATION_PROMPT

        tree_paths = None
        if topic_tree is not None:
            tree_paths = topic_tree.tree_paths
            total_paths = len(tree_paths)
            required_samples = num_steps * batch_size

            if required_samples > total_paths:
                raise ValueError(  # noqa: TRY003
                    f"Required samples ({required_samples}) exceeds available tree paths ({total_paths})"
                )  # noqa: TRY003

            tree_paths = random.sample(tree_paths, required_samples)
            num_steps = math.ceil(len(tree_paths) / batch_size)

        total_samples = num_steps * batch_size
        print(f"Generating dataset using model {self.model_name}")
        print(f"Generating dataset in {num_steps} steps, with batch size {batch_size}")

        # Enable JSON schema validation
        litellm.enable_json_schema_validation = True

        try:
            with tqdm(total=total_samples, desc="Progress") as pbar:
                for step in range(num_steps):
                    prompts = []
                    start_idx = step * batch_size

                    for i in range(batch_size):
                        path = None
                        if tree_paths:
                            current_idx = start_idx + i
                            if current_idx < len(tree_paths):
                                path = tree_paths[current_idx]
                            else:
                                break

                        sample_prompt = self.build_prompt(
                            data_creation_prompt=data_creation_prompt,
                            num_example_demonstrations=num_example_demonstrations,
                            subtopics_list=path,
                        )
                        prompts.append(sample_prompt)

                    for attempt in range(self.args.max_retries):
                        try:
                            responses = litellm.batch_completion(
                                model=self.model_name,
                                messages=[
                                    [{"role": "user", "content": p}] for p in prompts
                                ],
                                temperature=self.args.temperature,
                            )

                            samples = []
                            for r in responses:
                                response_content = r.choices[0].message.content
                                parsed_json = validate_json_response(response_content)

                                if parsed_json and include_sys_msg:
                                    # Add system message at the start if sys_msg is True
                                    if "messages" in parsed_json:
                                        parsed_json["messages"].insert(0, {
                                            "role": "system",
                                            "content": self.original_system_prompt
                                        })

                                if parsed_json:
                                    samples.append(parsed_json)
                                else:
                                    self.failed_samples.append(response_content)
                                    failure_type = self.analyze_failure(
                                        response_content
                                    )
                                    self.failure_analysis[failure_type].append(
                                        response_content
                                    )

                            if samples:
                                failed_samples, failure_descriptions = (
                                    self.dataset.add_samples(samples)
                                )
                                if failed_samples:
                                    for sample, desc in zip(
                                        failed_samples,
                                        failure_descriptions,
                                        strict=True,
                                    ):
                                        self.failed_samples.append(sample)
                                        self.failure_analysis["invalid_schema"].append(
                                            desc
                                        )
                                pbar.update(len(samples) - len(failed_samples))
                                break  # Success - exit retry loop

                        except Exception as e:
                            if attempt == self.args.max_retries - 1:
                                print(
                                    f"Failed after {self.args.max_retries} attempts: {str(e)}"
                                )
                                self.failed_samples.append(str(e))
                                failure_type = self.analyze_failure(str(e), error=e)
                                self.failure_analysis[failure_type].append(str(e))
                            else:
                                print(f"Attempt {attempt + 1} failed: {str(e)}")

        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
            self.print_failure_summary()
            self.save_dataset("interrupted_dataset.jsonl")
            return self.dataset

        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            self.print_failure_summary()
            self.save_dataset("error_dataset.jsonl")
            raise

        print(f"Successfully Generated {len(self.dataset)} samples.")
        self.print_failure_summary()
        return self.dataset

    def print_failure_summary(self):
        """Print a detailed summary of all failures."""
        summary = self.summarize_failures()

        print("\n=== Failure Analysis Summary ===")
        print(f"Total Failed Samples: {summary['total_failures']}")
        print("\nFailure Types Breakdown:")
        for failure_type, count in summary["failure_types"].items():
            if count > 0:
                print(f"\n{failure_type.replace('_', ' ').title()}: {count}")
                if failure_type in summary["failure_examples"]:
                    print("Example failures:")
                    for i, example in enumerate(
                        summary["failure_examples"][failure_type], 1
                    ):
                        print(f"  {i}. {example}")
        print("\n=============================")

    def build_prompt(
        self,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        subtopics_list: list[str] = None,
    ) -> str:
        prompt = data_creation_prompt.replace(
            "{{{{system_prompt}}}}", self.generation_system_prompt
        )
        prompt = prompt.replace(
            "{{{{instructions}}}}", self.build_custom_instructions_text()
        )
        prompt = prompt.replace(
            "{{{{examples}}}}", self.build_examples_text(num_example_demonstrations)
        )
        return prompt.replace(
            "{{{{subtopics}}}}", self.build_subtopics_text(subtopics_list)
        )

    def build_system_prompt(self):
        """Return the original system prompt for dataset inclusion."""
        return self.original_system_prompt

    def build_custom_instructions_text(self) -> str:
        if self.args.instructions is None:
            return ""
        return f"\nHere are additional instructions:\n<instructions>\n{self.args.instructions}\n</instructions>\n"

    def build_examples_text(self, num_example_demonstrations: int):
        if self.args.example_data is None or num_example_demonstrations == 0:
            return ""

        examples = random.sample(
            self.args.example_data.samples, num_example_demonstrations
        )
        examples_text = "Here are output examples:\n\n"
        examples_text += "\n".join(
            f"Example {i+1}: \n\n{ex}\n" for i, ex in enumerate(examples)
        )
        return (
            f"\nHere are output examples:\n<examples>\n{examples_text}\n</examples>\n"
        )

    def build_subtopics_text(self, subtopic_list: list[str]):
        if subtopic_list is None:
            return ""
        return f"\nLastly, the topic of the training data should be related to the following subtopics: {' -> '.join(subtopic_list)}"

    def save_dataset(self, save_path: str):
        """Save the dataset to a file."""
        self.dataset.save(save_path)
