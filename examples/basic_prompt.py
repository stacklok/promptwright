import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import LocalDataEngine, LocalEngineArguments


def main():
    print("promptwright - Synthetic Dataset Generation")
    print("==========================================")

    # First verify Ollama connection
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate a simple test response.",
            system_prompt="You are a helpful assistant.",
            model_name="mistral:latest",
            temperature=0.7,
            max_retries=2,
            prompt_template="""Return this exact JSON structure with a simple question and answer:
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. You provide clear and concise answers to user questions."
                    },
                    {
                        "role": "user",
                        "content": "What is 2+2?"
                    },
                    {
                        "role": "assistant",
                        "content": "2+2 equals 4."
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

        # Generate a single test sample with system message consistency
        # Amend the number of steps and batch size as needed, the current values
        # are set for a single test sample
        print("\nGenerating test sample(s)...")
        dataset = engine.create_data(num_steps=1, batch_size=1, topic_tree=None)

        # Ensure consistency by adding the system message explicitly
        for data in dataset:
            if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                data["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. You provide clear and concise answers to user questions.",
                    },
                )

        if len(dataset) > 0:
            print("\nTest successful! Starting main generation...")

            # Ask for confirmation before proceeding with full generation
            response = input("\nProceed with full generation? (y/n): ")
            if response.lower() == "y":
                dataset = engine.create_data(num_steps=100, batch_size=10, topic_tree=None)

                # Ensure system message consistency in all data
                for data in dataset:
                    if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                        data["messages"].insert(
                            0,
                            {
                                "role": "system",
                                "content": "You are a helpful assistant. You provide clear and concise answers to user questions.",
                            },
                        )

                dataset.save("full_dataset.jsonl")

        else:
            print("\nError: Test generation failed")

    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Try restarting Ollama: 'killall ollama && ollama serve'")
        print("2. Verify model is installed: 'ollama pull $model_name'")
        print("3. Check Ollama logs for errors")
        raise


if __name__ == "__main__":
    main()
