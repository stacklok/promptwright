import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = """You are a culinary expert who documents recipes and cooking techniques.
Your entries should be detailed, precise, and include both traditional and modern cooking methods."""

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Global Cuisine and Cooking Techniques", # Root prompt for the tree
        model_system_prompt=system_prompt, # System prompt for the model
        tree_degree=5,  # Different cuisine types
        tree_depth=3,   # Specific dishes and techniques
        temperature=0.7, # Balanced temperature for creativity and precision
        model_name="ollama/llama3" # Model name
    )
)

tree.build_tree()
tree.save("culinary_techniques_tree.jsonl")

engine = DataEngine(
    args=EngineArguments(
        instructions="""Create detailed recipe and technique entries that include:
                       - Ingredient lists with possible substitutions
                       - Step-by-step instructions
                       - Critical technique explanations
                       - Common mistakes to avoid
                       - Storage and serving suggestions
                       - Cultural context and history""", # Instructions for the model
        system_prompt=system_prompt, # System prompt for the model
        model_name="ollama/llama3", # Model name
        temperature=0.1,  # Balance between creativity and precision
        max_retries=3 # Retry failed prompts up to 3 times
    )
)

dataset = engine.create_data(
    num_steps=15, # Generate 15 entries
    batch_size=2, # Generate 2 entries at a time
    topic_tree=tree,
)

dataset.save("culinary_database.jsonl")
