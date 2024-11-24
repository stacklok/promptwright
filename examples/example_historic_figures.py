import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = """You are a knowledgeable historian who creates detailed, accurate biographical entries.
Each entry should include: birth/death dates, major achievements, historical impact, and interesting anecdotes."""

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Notable Historical Figures Across Different Eras and Fields",
        model_system_prompt=system_prompt,
        tree_degree=4,  # More branches for different categories
        tree_depth=3,  # Deeper tree for more specific figures
        temperature=0.6,  # Balanced temperature for creativity and accuracy
        model_name="ollama/llama3",  # Model name
    )
)

tree.build_tree()
tree.save("historical_figures_tree.jsonl")

engine = DataEngine(
    args=EngineArguments(
        instructions="""Generate biographical entries for historical figures.
                       Include lesser-known details and focus on their lasting impact.
                       Each entry should be engaging while maintaining historical accuracy.""",  # Instructions for the model
        system_prompt=system_prompt,  # System prompt for the model
        model_name="ollama/llama3",  # Model name
        temperature=0.7,  # Balance between creativity and accuracy
        max_retries=3,  # Retry failed generations up to 3 times
    )
)


dataset = engine.create_data(
    num_steps=15,  # Generate 15 entries
    batch_size=2,  # Generate 2 entries at a time
    topic_tree=tree,
)

dataset.save("historical_figures_database.jsonl")
