import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = """You are a creative writing instructor providing writing prompts and example responses. If you use apostrophes in your prompts, make sure to escape them with a backslash. For example, use 'don\'t' instead of 'don't'. Respond only with valid JSON. Do not write an introduction or summary."""

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Creative Writing Prompts",
        model_system_prompt=system_prompt,
        tree_degree=5,
        tree_depth=4,
        temperature=0.7,
        model_name="ollama/llama3",
    )
)

tree.build_tree()
tree.save("numpy_topictree.jsonl")

# Initialize engine with creative writing-specific parameters
engine = DataEngine(
    args=EngineArguments(
        instructions="Generate creative writing prompts and example responses.",  # Instructions for the model
        system_prompt=system_prompt,  # System prompt for the model
        model_name="ollama/llama3",  # Model name
        temperature=0.7,  # Higher temperature for creative writing
        max_retries=3,  # Retry failed prompts up to 3 times
    )
)

dataset = engine.create_data(
    num_steps=10,
    batch_size=5,
    topic_tree=tree,
    model_name="ollama/llama3.2",
)

dataset.save("creative_writing.jsonl")
