import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = """You are an expert programming instructor who creates engaging coding challenges.
Each challenge should test specific programming concepts while remaining accessible and educational."""

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Programming Challenges Across Different Difficulty Levels and Concepts",  # Root prompt for the tree
        model_system_prompt=system_prompt,  # System prompt for the model
        tree_degree=4,  # Different programming concepts
        tree_depth=2,  # Various difficulty levels
        temperature=0.7,  # Higher temperature for creative problem scenarios
        model_name="ollama/llama3",  # Model name
    )
)

tree.build_tree()
tree.save("programming_challenges_tree.jsonl")

engine = DataEngine(
    args=EngineArguments(
        instructions="""Generate programming challenges that include:
                       - Problem description
                       - Input/Output examples
                       - Constraints and edge cases
                       - Hint system (progressive hints)
                       - Solution approach discussion
                       - Time/Space complexity requirements""",  # Instructions for the model
        system_prompt=system_prompt,  # System prompt for the model
        model_name="ollama/llama3",  # Model name
        temperature=0.8,  # Higher temperature for creative problem scenarios
        max_retries=3,  # Retry failed generations up to 3 times
    )
)

dataset = engine.create_data(
    num_steps=6,
    batch_size=2,
    topic_tree=tree,
)

dataset.save("programming_challenges.jsonl")
