import ast
import json

from dataclasses import dataclass

from .ollama_client import OllamaClient


@dataclass
class LocalTopicTreeArguments:
    root_prompt: str
    model_system_prompt: str = None
    tree_degree: int = 10
    tree_depth: int = 3
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "llama3.2"


class LocalTopicTree:
    def __init__(self, args: LocalTopicTreeArguments):
        self.args = args
        self.tree_paths = []
        self.llm_client = OllamaClient(base_url=args.ollama_base_url)

    def _extract_list_from_response(self, response: str) -> list[str]:
        """Extract a Python list from the response text, with multiple fallback methods."""
        # First, try to find a list in JSON format
        try:
            # Try to parse as JSON first
            data = json.loads(response)
            if isinstance(data, list):
                return data
            # If it's a JSON object, look for a list value
            for value in data.values():
                if isinstance(value, list):
                    return value
        except json.JSONDecodeError:
            pass

        # Second, try to find and parse a Python list literal
        try:
            # Find content between square brackets
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1:
                list_str = response[start : end + 1]
                return ast.literal_eval(list_str)
        except (SyntaxError, ValueError):
            pass

        # Third, try to split by commas if the response looks like a comma-separated list
        if "," in response and "[" not in response and "{" not in response:
            try:
                items = [item.strip().strip("\"'") for item in response.split(",")]
                if items:
                    return items
            except Exception as e:
                print(f"Error parsing comma-separated list: {str(e)}")

        raise ValueError(f"Could not extract list from response: {response}")  # noqa: TRY003

    def build_tree(self):
        """Build the topic tree."""
        print(
            f"\nBuilding topic tree with degree {self.args.tree_degree} and depth {self.args.tree_depth}"
        )
        self.tree_paths = self.build_subtree(
            [self.args.root_prompt],
            self.args.model_system_prompt,
            self.args.tree_degree,
            self.args.tree_depth,
        )
        print(f"Tree building complete. Generated {len(self.tree_paths)} paths.")

    def build_subtree(
        self, node_path: list[str], system_prompt: str, tree_degree: int, subtree_depth: int
    ) -> list[list[str]]:
        """Build a subtree recursively."""
        print(f"Building subtree for path: {' -> '.join(node_path)}")

        if subtree_depth == 0:
            return [node_path]

        try:
            subnodes = self.get_subtopics(
                system_prompt=system_prompt, node_path=node_path, num_subtopics=tree_degree
            )

            if not subnodes:
                print(f"Warning: No subtopics generated for path: {' -> '.join(node_path)}")
                return [node_path]

            updated_node_paths = [node_path + [sub] for sub in subnodes]
            result = []

            for path in updated_node_paths:
                result.extend(
                    self.build_subtree(path, system_prompt, tree_degree, subtree_depth - 1)
                )
            return result  # noqa: TRY300

        except Exception as e:
            print(f"Error building subtree for path {' -> '.join(node_path)}: {str(e)}")
            return [node_path]

    def get_subtopics(
        self, system_prompt: str, node_path: list[str], num_subtopics: int
    ) -> list[str]:
        """Get subtopics for a given node."""
        prompt = f"""Generate exactly {num_subtopics} subtopics about: {' -> '.join(node_path)}

Requirements:
1. Return ONLY a Python list of strings
2. Each subtopic should be short and focused
3. No explanations or additional text
4. No numbered bullets or formatting
5. No nested lists or dictionaries

Example output format:
["Subtopic 1", "Subtopic 2", "Subtopic 3"]

Generate {num_subtopics} subtopics:"""

        try:
            response = self.llm_client.generate_completion(
                prompt=prompt,
                model=self.args.model_name,
                system_prompt=system_prompt
                or "You are a helpful assistant that generates lists of subtopics.",
                temperature=0.7,
            )

            subtopics = self._extract_list_from_response(response.content)

            # Validate and clean the subtopics
            cleaned_subtopics = []
            for topic in subtopics:
                if isinstance(topic, str):
                    # Remove any quotes, brackets, or list formatting
                    cleaned = topic.strip(" []\"'")
                    if cleaned:
                        cleaned_subtopics.append(cleaned)

            # Ensure we have the right number of subtopics
            if len(cleaned_subtopics) < num_subtopics:
                print(
                    f"Warning: Only generated {len(cleaned_subtopics)} subtopics instead of {num_subtopics}"
                )

            return cleaned_subtopics[:num_subtopics]

        except Exception as e:
            print(f"Error generating subtopics: {str(e)}")
            print(f"Response content: {getattr(response, 'content', 'No content available')}")
            raise

    def save(self, save_path: str):
        """Save the topic tree to a file."""
        with open(save_path, "w") as f:
            for path in self.tree_paths:
                f.write(json.dumps({"path": path}) + "\n")
        print(f"\nTopic tree saved to {save_path}")
        print(f"Total paths: {len(self.tree_paths)}")

    def print_tree(self):
        """Print the topic tree in a readable format."""
        print("\nTopic Tree Structure:")
        for path in self.tree_paths:
            print(" -> ".join(path))
