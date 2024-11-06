import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments, TopicTree, TopicTreeArguments

system_prompt = """You are a red team exercise designer creating advanced cybersecurity training scenarios.
Focus on creating realistic, detailed scenarios that help organizations improve their security posture.
All scenarios must emphasize legal compliance and ethical considerations."""

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Advanced Red Team Exercise Scenarios and Methodologies",  # Root prompt for the tree
        model_system_prompt=system_prompt,
        tree_degree=4,  # Different attack vectors
        tree_depth=4,  # Detailed scenario branches
        temperature=0.6,  # Balanced for creativity within constraints
        model_name="ollama/qwen2.5",  # Model name
    )
)

tree.build_tree()
tree.save("redteam_scenarios_tree.jsonl")

engine = DataEngine(
    args=EngineArguments(
        instructions="""Generate red team training scenarios. Each scenario should be realistic and focus on improving
organizational security posture while maintaining ethical standards.""",  # Instructions for the model
        system_prompt=system_prompt,  # System prompt for the model
        model_name="ollama/llama3",  # Model name
        temperature=0.6,  # Balanced for creativity within constraints
        max_retries=3,  # Retry failed prompts up to 3 times
    )
)

dataset = engine.create_data(
    num_steps=12,  # Generate 12 scenarios
    batch_size=2,  # Generate 2 scenarios at a time
    topic_tree=tree,  # Use the red team scenarios tree
    model_name="ollama/qwen2.5",  # Use the Qwen model
)

dataset.save("redteam_scenarios_dataset.jsonl")
