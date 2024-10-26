import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import LocalDataEngine, LocalEngineArguments


def main():
    print("promptwright - Programming Dataset Generation")
    print("=====================================")

    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate a simple programming question and answer.",
            system_prompt="You are a programming expert.",
            model_name="llama3.2",
            temperature=0.7,
            max_retries=2,
            prompt_template="""Return this exact JSON structure with a programming question and answer:
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a programming expert. You provide clear code examples and explanations."
                    },
                    {
                        "role": "user",
                        "content": "How do I print Hello World in Python?"
                    },
                    {
                        "role": "assistant",
                        "content": "To print Hello World in Python, use this code:\\n```python\\nprint('Hello World')\\n```\\nThis will display the text Hello World in the console."
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
        print("\nGenerating test sample...")
        dataset = engine.create_data(num_steps=1, batch_size=1, topic_tree=None)

        # Ensure consistency by adding the system message explicitly
        for data in dataset:
            if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                data["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": "You are a programming instructor. You provide clear code examples and explanations.",
                    },
                )

        if len(dataset) > 0:
            print("\nTest successful! Starting main generation...")
            response = input("\nProceed with full generation? (y/n): ")
            if response.lower() == "y":
                dataset = engine.create_data(num_steps=100, batch_size=10, topic_tree=None)

                # Ensure system message consistency
                for data in dataset:
                    if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                        data["messages"].insert(
                            0,
                            {
                                "role": "system",
                                "content": "You are a programming instructor. You provide clear code examples and explanations.",
                            },
                        )

                dataset.save("programming_dataset.jsonl")
                print(f"\nSaved {len(dataset)} programming Q&A pairs to programming_dataset.jsonl")
        else:
            print("\nError: Test generation failed")

    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Try restarting Ollama: 'killall ollama && ollama serve'")
        print("2. Verify model is installed: 'ollama pull llama3.2'")
        print("3. Check Ollama logs for errors")
        raise


if __name__ == "__main__":
    main()
