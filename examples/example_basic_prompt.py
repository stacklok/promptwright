import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = "You are a helpful assistant. You provide clear and concise answers to user questions."

tree = TopicTree(
        args=TopicTreeArguments(
            root_prompt="Capital Cities of the World.",
            model_system_prompt=system_prompt,
            tree_degree=3, # Different continents
            tree_depth=2, # Deeper tree for more specific topics
            temperature=0.7, # Higher temperature for more creative variations
            model_name="ollama/mistral:latest" # Model name
        )
)

tree.build_tree()
tree.save("basic_prompt_topictree.jsonl")

engine = DataEngine(
    args=EngineArguments(
        instructions="Please provide training examples with questions about capital cities of the world.", # Instructions for the model
        system_prompt=system_prompt, # System prompt for the model
        model_name="ollama/mistral:latest", # Model name
        temperature=0.9,  # Higher temperature for more creative variations
        max_retries=2, # Retry failed prompts up to 2 times
    )
)

dataset = engine.create_data(
    num_steps=5,
    batch_size=1,
    topic_tree=tree,
    model_name="ollama/mistral:latest",
)

dataset.save("basic_prompt_dataset.jsonl")
