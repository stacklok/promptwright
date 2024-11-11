import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments  # noqa: E402

system_prompt = (
    "You are a helpful assistant. You provide clear and concise answers to user questions."
)
engine = DataEngine(
    args=EngineArguments(
        instructions="Please provide training examples with questions about capital cities of the world.",  # Instructions for the model
        system_prompt=system_prompt,  # System prompt for the model
        model_name="ollama/mistral:latest",  # Model name
        temperature=0.9,  # Higher temperature for more creative variations
        max_retries=2,  # Retry failed prompts up to 2 times
    )
)

questions = engine.create_questions(10)
print("Questions are", questions)
