from datasets import load_dataset
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


class HFUploader:
    """
    HFUploader is a class for uploading datasets to the Hugging Face Hub.

    Methods
    -------
    __init__(hf_token)

    push_to_hub(hf_dataset_repo, jsonl_file_path)

        Parameters
        ----------
        hf_dataset_repo : str
            The repository name in the format 'username/dataset_name'.
        jsonl_file_path : str
            Path to the JSONL file.

        Returns
        -------
        dict
            A dictionary containing the status and a message.
    """
    def __init__(self, hf_token):
        """
        Initialize the uploader with the Hugging Face authentication token.

        Parameters:
        hf_token (str): Hugging Face Hub authentication token.
        """
        self.hf_token = hf_token

    def push_to_hub(self, hf_dataset_repo, jsonl_file_path):
        """
        Push a JSONL dataset to Hugging Face Hub.

        Parameters:
        hf_dataset_repo (str): The repository name in the format 'username/dataset_name'.
        jsonl_file_path (str): Path to the JSONL file.

        Returns:
        dict: A dictionary containing the status and a message.
        """
        try:
            login(token=self.hf_token)
            dataset = load_dataset("json", data_files={"train": jsonl_file_path})
            dataset.push_to_hub(hf_dataset_repo, token=self.hf_token)

        except RepositoryNotFoundError:
            return {
                "status": "error",
                "message": f"Repository '{hf_dataset_repo}' not found. Please check your repository name.",
            }

        except HfHubHTTPError as e:
            return {"status": "error", "message": f"Hugging Face Hub HTTP Error: {str(e)}"}

        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"File '{jsonl_file_path}' not found. Please check your file path.",
            }

        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

        else:
            return {
                "status": "success",
                "message": f"Dataset pushed successfully to {hf_dataset_repo}.",
            }
