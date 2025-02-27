from my_furhat_backend.config.settings import config, BASE_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from huggingface_hub.errors import GatedRepoError
import torch
import os

class ModelPipelineManager:
    """
    Manages the creation and retrieval of a Hugging Face text-generation pipeline.

    This class handles pipeline instantiation for causal language models. It automatically 
    manages model loading, including authentication for gated repositories, and sets default 
    generation parameters.
    """

    def __init__(self):
        """
        Initialize the ModelPipelineManager with a placeholder for the pipeline.
        """
        self._pipeline = None

    def _create_pipeline(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        **kwargs
    ):
        """
        Create a Hugging Face text-generation pipeline using the specified model.

        This method sets default generation parameters (max_new_tokens, top_k, temperature)
        if they are not provided. It attempts to load the tokenizer and model, handling potential
        authentication issues for gated repositories by re-logging into Hugging Face if needed.

        Parameters:
            model_id (str): The Hugging Face model identifier to load.
            **kwargs: Additional generation parameters to pass to the pipeline. 
                      Defaults include:
                          - max_new_tokens: Maximum number of new tokens to generate (default: 100)
                          - top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering (default: 50)
                          - temperature: The value used to module the next token probabilities (default: 0.1)

        Returns:
            A Hugging Face pipeline object configured for text generation.
        """
        # Set default generation parameters if not provided
        kwargs.setdefault("max_new_tokens", 100)
        kwargs.setdefault("top_k", 50)
        kwargs.setdefault("temperature", 0.1)

        try:
            # Attempt to load the tokenizer and model from the given model ID.
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            # Check for authentication errors related to gated repositories.
            if ("gated repo" in str(e)) or ("Unauthorized" in str(e)) or isinstance(e, GatedRepoError):
                print("Authentication error detected. Re-logging into Hugging Face...")
                # Retrieve the Hugging Face token from the configuration.
                hf_token = config["HF_KEY"]
                if not hf_token:
                    raise RuntimeError("HF_KEY environment variable is not set.")
                # Log in to Hugging Face with the provided token.
                login(token=hf_token)
                # Retry loading the tokenizer and model after authentication.
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Reraise any other exceptions that occur.
                raise e

        # Create and return the text-generation pipeline using the loaded model and tokenizer.
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            **kwargs
        )

    def get_pipeline(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", **kwargs):
        """
        Retrieve the text-generation pipeline. If not already created, instantiate it.

        Parameters:
            model_id (str): The model identifier to use for creating the pipeline (default: "mistralai/Mistral-7B-Instruct-v0.3").
            **kwargs: Additional generation parameters to pass during pipeline creation.

        Returns:
            A Hugging Face text-generation pipeline.
        """
        # Create the pipeline only if it hasn't been created yet.
        if self._pipeline is None:
            self._pipeline = self._create_pipeline(model_id, **kwargs)
        return self._pipeline
