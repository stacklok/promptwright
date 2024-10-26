import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import LocalDataEngine, LocalEngineArguments


def main():
    print("promptwright - Creative Writing Dataset Generation")
    print("=====================================")

    # Initialize engine with creative writing-specific parameters
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate creative writing prompts and example responses.",
            system_prompt="You are a creative writing instructor providing writing prompts and example responses.",
            model_name="llama3.2",
            temperature=0.9,  # Higher temperature for more creative variations
            max_retries=2,
            prompt_template="""Return this exact JSON structure with a writing prompt and creative response:
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a creative writing instructor providing writing prompts and example responses."
                    },
                    {
                        "role": "user",
                        "content": "Write a short story about finding an unexpected letter."
                    },
                    {
                        "role": "assistant",
                        "content": "The envelope was weathered, its corners soft from age. Sarah found it tucked behind the loose baseboard she'd been meaning to fix, addressed in her grandmother's distinctive cursive. The postmark read 1962—this letter had waited sixty years to be found. With trembling fingers, she broke the seal..."
                    }
                ]
            }""",
        )
    )

    try:
        # Test Ollama connection
        print("\nTesting Ollama connection...")
        models = engine.llm_client.list_local_models()
        print(f"Available models: {[m['name'] for m in models]}")

        # Generate a single test sample
        print("\nGenerating writing test sample...")
        dataset = engine.create_data(num_steps=1, batch_size=1, topic_tree=None)

        # Ensure consistency by adding the system message explicitly
        for data in dataset:
            if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                data["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": "You are a creative writing instructor providing writing prompts and example responses.",
                    },
                )

        if len(dataset) > 0:
            print("\nTest successful! Starting main generation...")
            response = input("\nProceed with full writing dataset generation? (y/n): ")
            if response.lower() == "y":
                dataset = engine.create_data(num_steps=100, batch_size=10, topic_tree=None)

                # Ensure system message consistency
                for data in dataset:
                    if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                        data["messages"].insert(
                            0,
                            {
                                "role": "system",
                                "content": "You are a creative writing instructor providing writing prompts and example responses.",
                            },
                        )

                dataset.save("writing_dataset.jsonl")
                print(
                    f"\nSaved {len(dataset)} writing prompts and responses to writing_dataset.jsonl"
                )
        else:
            print("\nError: Test generation failed")

    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Try restarting Ollama: 'killall ollama && ollama serve'")
        print("2. Verify model is installed: 'ollama pull mistral:latest'")
        print("3. Check Ollama logs for errors")
        raise


if __name__ == "__main__":
    main()
