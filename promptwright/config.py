"""Configuration handling for YAML-based promptwright configurations."""

import os
from dataclasses import dataclass

import yaml

from .engine import EngineArguments
from .topic_tree import TopicTreeArguments


def construct_model_string(provider: str, model: str) -> str:
    """Construct the full model string for LiteLLM."""
    return f"{provider}/{model}"


@dataclass
class PromptWrightConfig:
    """Configuration for PromptWright tasks."""

    system_prompt: str
    topic_tree: dict
    data_engine: dict
    dataset: dict
    huggingface: dict | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PromptWrightConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            system_prompt=config_dict.get("system_prompt", ""),
            topic_tree=config_dict.get("topic_tree", {}),
            data_engine=config_dict.get("data_engine", {}),
            dataset=config_dict.get("dataset", {}),
            huggingface=config_dict.get("huggingface", None),
        )

    def get_topic_tree_args(self, **overrides) -> TopicTreeArguments:
        """Get TopicTreeArguments from config with optional overrides."""
        args = self.topic_tree.get("args", {}).copy()  # Make a copy of the args

        # Replace system prompt placeholder
        if "model_system_prompt" in args:
            args["model_system_prompt"] = args["model_system_prompt"].replace(
                "<system_prompt_placeholder>", self.system_prompt
            )

        # Handle provider and model separately
        provider = overrides.pop("provider", args.pop("provider", "ollama"))
        model = overrides.pop("model", args.pop("model", "mistral:latest"))

        # Apply remaining overrides
        args.update(overrides)

        # Construct full model string
        args["model_name"] = construct_model_string(provider, model)

        return TopicTreeArguments(
            root_prompt=args.get("root_prompt", ""),
            model_system_prompt=args.get("model_system_prompt", ""),
            tree_degree=args.get("tree_degree", 3),
            tree_depth=args.get("tree_depth", 2),
            temperature=args.get("temperature", 0.7),
            model_name=args["model_name"],
        )

    def get_engine_args(self, **overrides) -> EngineArguments:
        """Get EngineArguments from config with optional overrides."""
        args = self.data_engine.get("args", {}).copy()  # Make a copy of the args

        # Replace system prompt placeholder
        if "system_prompt" in args:
            args["system_prompt"] = args["system_prompt"].replace(
                "<system_prompt_placeholder>", self.system_prompt
            )

        # Handle provider and model separately
        provider = overrides.pop("provider", args.pop("provider", "ollama"))
        model = overrides.pop("model", args.pop("model", "mistral:latest"))

        # Apply remaining overrides
        args.update(overrides)

        # Construct full model string
        args["model_name"] = construct_model_string(provider, model)

        # Get sys_msg from dataset config, defaulting to True
        dataset_config = self.get_dataset_config()
        sys_msg = dataset_config.get("creation", {}).get("sys_msg", True)

        return EngineArguments(
            instructions=args.get("instructions", ""),
            system_prompt=args.get("system_prompt", ""),
            model_name=args["model_name"],
            temperature=args.get("temperature", 0.9),
            max_retries=args.get("max_retries", 2),
            sys_msg=sys_msg,
        )

    def get_dataset_config(self) -> dict:
        """Get dataset configuration."""
        return self.dataset

    def get_huggingface_config(self) -> dict:
        """Get Hugging Face configuration."""
        return self.huggingface or {}
