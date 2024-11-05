import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = """You are a scientific database curator specializing in experimental procedures and results.
Your role is to document experiments clearly, including methodology, materials, observations, and conclusions."""

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Groundbreaking Scientific Experiments Throughout History", # Root prompt for the tree
        model_system_prompt=system_prompt, # System prompt for the model
        tree_degree=3,  # Branch into different scientific fields
        tree_depth=3,   # Go deeper into specific experiments
        temperature=0.6, # Lower temperature for more precise content
        model_name="ollama/llama3"
    )
)

tree.build_tree()
tree.save("science_experiments_tree.jsonl") # Save the generated topic tree to a file

engine = DataEngine(
    args=EngineArguments(
        instructions="""Create detailed experimental procedure entries.
                       Each entry should include:
                       - Required materials and equipment
                       - Step-by-step methodology
                       - Expected results and observations
                       - Common pitfalls and troubleshooting
                       - Safety considerations""", # Instructions for the model
        system_prompt=system_prompt, # System prompt for the model
        model_name="ollama/llama3", # Model name
        temperature=0.4,  # Lower temperature for more precise content
        max_retries=3, # Retry failed prompts up to 3 times
    )
)

dataset = engine.create_data(
    num_steps=8, # Number of steps to generate
    batch_size=2, # Batch size for each step
    topic_tree=tree, # Topic tree used to guide the generation
)

dataset.save("scientific_experiments.jsonl") # Save the generated dataset to a file
