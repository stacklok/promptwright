"""Tests for the CLI module."""

import os
import tempfile

from unittest.mock import Mock, patch

import pytest

from click.testing import CliRunner

from promptwright.cli import cli


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return """
system_prompt: "Test system prompt"
topic_tree:
  args:
    root_prompt: "Test root prompt"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 3
    tree_depth: 2
    temperature: 0.7
    provider: "test"
    model: "model"
  save_as: "test_tree.jsonl"
data_engine:
  args:
    instructions: "Test instructions"
    system_prompt: "<system_prompt_placeholder>"
    provider: "test"
    model: "model"
    temperature: 0.9
    max_retries: 2
dataset:
  creation:
    num_steps: 5
    batch_size: 1
    provider: "test"
    model: "model"
  save_as: "test_dataset.jsonl"
"""


@pytest.fixture
def sample_config_file(sample_yaml_content):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "PromptWright CLI" in result.output


def test_start_help(cli_runner):
    """Test start command help."""
    result = cli_runner.invoke(cli, ["start", "--help"])
    assert result.exit_code == 0
    assert "Generate training data from a YAML configuration file" in result.output


@patch("promptwright.cli.TopicTree")
@patch("promptwright.cli.DataEngine")
def test_start_command_basic(
    mock_data_engine, mock_topic_tree, cli_runner, sample_config_file
):
    """Test basic start command execution."""
    # Setup mocks
    mock_tree_instance = Mock()
    mock_engine_instance = Mock()
    mock_dataset = Mock()

    mock_topic_tree.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data.return_value = mock_dataset

    # Run command
    result = cli_runner.invoke(cli, ["start", sample_config_file])

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify mocks were called correctly
    mock_topic_tree.assert_called_once()
    mock_tree_instance.build_tree.assert_called_once()
    mock_tree_instance.save.assert_called_once()
    mock_data_engine.assert_called_once()
    mock_engine_instance.create_data.assert_called_once()
    mock_dataset.save.assert_called_once()


@patch("promptwright.cli.TopicTree")
@patch("promptwright.cli.DataEngine")
def test_start_command_with_overrides(
    mock_data_engine, mock_topic_tree, cli_runner, sample_config_file
):
    """Test start command with parameter overrides."""
    # Setup mocks
    mock_tree_instance = Mock()
    mock_engine_instance = Mock()
    mock_dataset = Mock()

    mock_topic_tree.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data.return_value = mock_dataset

    # Run command with overrides
    result = cli_runner.invoke(
        cli,
        [
            "start",
            sample_config_file,
            "--topic-tree-save-as",
            "override_tree.jsonl",
            "--dataset-save-as",
            "override_dataset.jsonl",
            "--provider",
            "override",
            "--model",
            "model",
            "--temperature",
            "0.5",
            "--tree-degree",
            "4",
            "--tree-depth",
            "3",
            "--num-steps",
            "10",
            "--batch-size",
            "2",
        ],
    )

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify mocks were called with overridden values
    mock_topic_tree.assert_called_once()
    args, kwargs = mock_topic_tree.call_args
    assert kwargs["args"].model_name == "override/model"
    assert kwargs["args"].temperature == 0.5  # noqa: PLR2004
    assert kwargs["args"].tree_degree == 4  # noqa: PLR2004
    assert kwargs["args"].tree_depth == 3  # noqa: PLR2004

    mock_tree_instance.save.assert_called_once_with("override_tree.jsonl")
    mock_dataset.save.assert_called_once_with("override_dataset.jsonl")

    args, kwargs = mock_engine_instance.create_data.call_args
    assert kwargs["num_steps"] == 10  # noqa: PLR2004
    assert kwargs["batch_size"] == 2  # noqa: PLR2004
    assert kwargs["model_name"] == "override/model"


def test_start_command_missing_config(cli_runner):
    """Test start command with missing config file."""
    result = cli_runner.invoke(cli, ["start", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_start_command_invalid_yaml(cli_runner):
    """Test start command with invalid YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_path = f.name

    try:
        result = cli_runner.invoke(cli, ["start", temp_path])
        assert result.exit_code != 0
        assert "Error" in result.output
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@patch("promptwright.cli.TopicTree")
@patch("promptwright.cli.DataEngine")
def test_start_command_error_handling(
    mock_data_engine, mock_topic_tree, cli_runner, sample_config_file  # noqa: ARG001
):
    """Test error handling in start command."""
    # Setup mock to raise an exception
    mock_topic_tree.side_effect = Exception("Test error")

    # Run command
    result = cli_runner.invoke(cli, ["start", sample_config_file])

    # Verify command failed with error
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "Test error" in result.output
