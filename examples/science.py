import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import LocalDataEngine, LocalEngineArguments


def main():
    print("promptwright - Scientific Dataset Generation")
    print("=====================================")

    # Initialize engine with science-specific parameters
    engine = LocalDataEngine(
        args=LocalEngineArguments(
            instructions="Generate scientific Q&A pairs covering physics, chemistry, and biology.",
            system_prompt="You are a science educator providing accurate, clear explanations.",
            model_name="llama3.2",
            temperature=0.7,
            max_retries=2,
            prompt_template="""Return this exact JSON structure with a scientific question and detailed answer:
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a science educator providing accurate, clear explanations."
                    },
                    {
                        "role": "user",
                        "content": "What causes the seasons on Earth?"
                    },
                    {
                        "role": "assistant",
                        "content": "Seasons are caused by Earth's tilted axis of rotation (23.5 degrees) as it orbits the Sun. This tilt means different hemispheres receive varying amounts of direct sunlight throughout the year, leading to seasonal changes in temperature and daylight hours."
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
        print("\nGenerating science test sample...")
        dataset = engine.create_data(num_steps=1, batch_size=1, topic_tree=None)

        # Ensure consistency by adding the system message explicitly
        for data in dataset:
            if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                data["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": "You are a science educator providing accurate, clear explanations.",
                    },
                )

        if len(dataset) > 0:
            print("\nTest successful! Starting main generation...")
            response = input("\nProceed with full science dataset generation? (y/n): ")
            if response.lower() == "y":
                dataset = engine.create_data(num_steps=100, batch_size=10, topic_tree=None)

                # Ensure system message consistency
                for data in dataset:
                    if not any(msg.get("role") == "system" for msg in data.get("messages", [])):
                        data["messages"].insert(
                            0,
                            {
                                "role": "system",
                                "content": "You are a science educator providing accurate, clear explanations.",
                            },
                        )

                dataset.save("science_dataset.jsonl")
                print(f"\nSaved {len(dataset)} science Q&A pairs to science_dataset.jsonl")
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
