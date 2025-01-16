import json
import re
import time
import warnings

from dataclasses import dataclass
from typing import Any

import litellm

from .prompts import TREE_GENERATION_PROMPT, TREE_JSON_INSTRUCTIONS
from .utils import extract_list

warnings.filterwarnings("ignore", message="Pydantic serializer warnings:.*")


def validate_and_clean_response(response_text: str) -> str | list[str] | None:
    """Clean and validate the response from the LLM."""
    try:
        # First try to extract a JSON array if present
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match:
            cleaned_json = json_match.group(0)
            # Remove any markdown code block markers
            cleaned_json = re.sub(r"```json\s*|\s*```", "", cleaned_json)
            return json.loads(cleaned_json)

        # If no JSON array found, fall back to extract_list
        topics = extract_list(response_text)
        if topics:
            return [topic.strip() for topic in topics if topic.strip()]
        return None  # noqa: TRY300
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing response: {str(e)}")
        return None


@dataclass
class TopicTreeArguments:
    """
    A class to represent the arguments for constructing a topic tree.

    Attributes:
        root_prompt (str): The initial prompt to start the topic tree.
        model_system_prompt (str): The system prompt for the model.
        tree_degree (int): The branching factor of the tree.
        tree_depth (int): The depth of the tree.
        model_name (str): The name of the model to be used.
    """

    root_prompt: str
    model_system_prompt: str = ""
    tree_degree: int = 10
    tree_depth: int = 3
    model_name: str = "ollama/llama3"
    temperature: float = 0.2


class TopicTreeValidator:
    """
    TopicTreeValidator validates and calculates unique paths in a tree structure.
    """

    def __init__(self, tree_degree: int, tree_depth: int):
        self.tree_degree = tree_degree
        self.tree_depth = tree_depth

    def calculate_paths(self) -> int:
        """Calculate total number of paths in the tree."""
        return self.tree_degree**self.tree_depth

    def validate_configuration(self, num_steps: int, batch_size: int) -> dict[str, Any]:
        """Validates tree configuration and provides recommendations if invalid."""
        total_requested_paths = num_steps * batch_size
        total_tree_paths = self.calculate_paths()

        print(f"Total tree paths available: {total_tree_paths}")
        print(f"Total requested paths: {total_requested_paths}")

        if total_requested_paths > total_tree_paths:
            print("Warning: The requested paths exceed the available tree paths.")
            recommendation = {
                "valid": False,
                "suggested_num_steps": total_tree_paths // batch_size,
                "suggested_batch_size": total_tree_paths // num_steps,
                "total_tree_paths": total_tree_paths,
                "total_requested_paths": total_requested_paths,
            }
            print("Recommended configurations to fit within the tree paths:")
            print(f" - Reduce num_steps to: {recommendation['suggested_num_steps']} or")
            print(
                f" - Reduce batch_size to: {recommendation['suggested_batch_size']} or"
            )
            print(" - Increase tree_depth or tree_degree to provide more paths.")
            return recommendation

        return {
            "valid": True,
            "total_tree_paths": total_tree_paths,
            "total_requested_paths": total_requested_paths,
        }


class TopicTree:
    """A class to represent and build a hierarchical topic tree."""

    def __init__(self, args: TopicTreeArguments):
        """Initialize the TopicTree with the given arguments."""
        if not args.model_name:
            raise ValueError(  # noqa: TRY003
                "model_name must be specified in TopicTreeArguments"
            )  # noqa: TRY003
        json_instructions = TREE_JSON_INSTRUCTIONS

        self.args = args
        self.system_prompt = json_instructions + args.model_system_prompt
        self.temperature = args.temperature
        self.model_name = args.model_name
        self.tree_degree = args.tree_degree
        self.tree_depth = args.tree_depth
        self.tree_paths = []
        self.failed_generations = []

    def build_tree(self, model_name: str = None) -> None:
        """Build the complete topic tree."""
        if model_name:
            self.model_name = model_name

        print(f"Building the topic tree with model: {self.model_name}")

        try:
            self.tree_paths = self.build_subtree(
                [self.args.root_prompt],
                self.system_prompt,
                self.args.tree_degree,
                self.args.tree_depth,
                model_name=self.model_name,
            )

            print(f"Tree building complete. Generated {len(self.tree_paths)} paths.")
            if self.failed_generations:
                print(
                    f"Warning: {len(self.failed_generations)} subtopic generations failed."
                )

        except Exception as e:
            print(f"Error building tree: {str(e)}")
            if self.tree_paths:
                print("Saving partial tree...")
                self.save("partial_tree.jsonl")
            raise

    def get_subtopics(
        self, system_prompt: str, node_path: list[str], num_subtopics: int
    ) -> list[str]:
        """Generate subtopics with improved error handling and validation."""
        print(f"Generating {num_subtopics} subtopics for: {' -> '.join(node_path)}")

        prompt = TREE_GENERATION_PROMPT
        prompt = prompt.replace(
            "{{{{system_prompt}}}}", system_prompt if system_prompt else ""
        )
        prompt = prompt.replace("{{{{subtopics_list}}}}", " -> ".join(node_path))
        prompt = prompt.replace("{{{{num_subtopics}}}}", str(num_subtopics))

        max_retries = 3
        retries = 0
        last_error = "No error recorded"

        while retries < max_retries:
            try:
                # Prepare completion arguments
                completion_args = {
                    "model": self.model_name,
                    "max_tokens": 1000,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }

                response = litellm.completion(**completion_args)

                subtopics = validate_and_clean_response(
                    response.choices[0].message.content
                )

                if subtopics and len(subtopics) > 0:
                    # Validate and clean each subtopic
                    cleaned_subtopics = []
                    for topic in subtopics:
                        if isinstance(topic, str):
                            # Keep more special characters but ensure JSON safety
                            cleaned_topic = topic.strip()
                            if cleaned_topic:
                                cleaned_subtopics.append(cleaned_topic)

                    if len(cleaned_subtopics) >= num_subtopics:
                        return cleaned_subtopics[:num_subtopics]

                last_error = "Insufficient valid subtopics generated"
                print(f"Attempt {retries + 1}: {last_error}. Retrying...")

            except Exception as e:
                last_error = str(e)
                print(
                    f"Error generating subtopics (attempt {retries + 1}/{max_retries}): {last_error}"
                )

            retries += 1
            if retries < max_retries:
                time.sleep(2**retries)  # Exponential backoff

        # If all retries failed, generate default subtopics and log the failure
        default_subtopics = [
            f"subtopic_{i+1}_for_{node_path[-1]}" for i in range(num_subtopics)
        ]
        self.failed_generations.append(
            {"path": node_path, "attempts": retries, "last_error": last_error}
        )
        print(
            f"Failed to generate valid subtopics after {max_retries} attempts. Using default subtopics."
        )
        return default_subtopics

    def build_subtree(
        self,
        node_path: list[str],
        system_prompt: str,
        tree_degree: int,
        subtree_depth: int,
        model_name: str,
    ) -> list[list[str]]:
        """Build a subtree with improved error handling and validation."""
        # Convert any non-string elements to strings
        node_path = [
            str(node) if not isinstance(node, str) else node for node in node_path
        ]
        print(f"Building topic subtree: {' -> '.join(node_path)}")

        if subtree_depth == 0:
            return [node_path]

        subnodes = self.get_subtopics(system_prompt, node_path, tree_degree)

        # Clean and validate subnodes
        cleaned_subnodes = []
        for subnode in subnodes:
            try:
                if isinstance(subnode, dict | list):
                    cleaned_subnodes.append(json.dumps(subnode))
                else:
                    cleaned_subnodes.append(str(subnode))
            except Exception as e:
                print(f"Error cleaning subnode: {str(e)}")
                continue

        result = []
        for subnode in cleaned_subnodes:
            try:
                new_path = node_path + [subnode]
                result.extend(
                    self.build_subtree(
                        new_path,
                        system_prompt,
                        tree_degree,
                        subtree_depth - 1,
                        model_name,
                    )
                )
            except Exception as e:
                print(f"Error building subtree for {subnode}: {str(e)}")
                continue

        return result

    def save(self, save_path: str) -> None:
        """Save the topic tree to a file."""
        try:
            with open(save_path, "w") as f:
                for path in self.tree_paths:
                    f.write(json.dumps({"path": path}) + "\n")

            # Save failed generations if any
            if self.failed_generations:
                failed_path = save_path.replace(".jsonl", "_failed.jsonl")
                with open(failed_path, "w") as f:
                    for failure in self.failed_generations:
                        f.write(json.dumps(failure) + "\n")
                print(f"Failed generations saved to {failed_path}")

            print(f"Topic tree saved to {save_path}")
            print(f"Total paths: {len(self.tree_paths)}")

        except Exception as e:
            print(f"Error saving topic tree: {str(e)}")
            raise

    def print_tree(self) -> None:
        """Print the topic tree in a readable format."""
        print("Topic Tree Structure:")
        for path in self.tree_paths:
            print(" -> ".join(path))

    def from_dict_list(self, dict_list: list[dict[str, Any]]) -> None:
        """
        Construct the topic tree from a list of dictionaries.

        Args:
            dict_list (list[dict]): The list of dictionaries representing the topic tree.
        """
        self.tree_paths = []
        self.failed_generations = []

        for d in dict_list:
            if 'path' in d:
                self.tree_paths.append(d['path'])
            if 'failed_generation' in d:
                self.failed_generations.append(d['failed_generation'])

        print(f"Loaded {len(self.tree_paths)} paths and {len(self.failed_generations)} failed generations from JSONL file")