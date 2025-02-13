# services/model_pipeline.py

from my_furhat_backend.config.settings import config, BASE_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from huggingface_hub.errors import GatedRepoError
import torch
import os

class ModelPipelineManager:
    def __init__(self):
        self._pipeline = None

    def _create_pipeline(
        self,
        model_id: str = "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        file_name: str = "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        **kwargs
    ):
        # Set default keyword arguments if not provided
        kwargs.setdefault("max_new_tokens", 100)
        kwargs.setdefault("top_k", 50)
        kwargs.setdefault("temperature", 0.1)

        # Optionally, you can convert file_name to an absolute path
        # if file_name is not None:
        #     absolute_file_path = os.path.abspath(file_name)
        #     if not os.path.exists(absolute_file_path):
        #         raise FileNotFoundError(f"GGUF file not found at path: {absolute_file_path}")
        # else:
        #     absolute_file_path = None

        try:
            # Use legacy=True if the checkpoint was built for a Bart-based tokenizer.
            tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=file_name, legacy=True)
            # For the model, do not pass the gguf_file if it's not supported.
            # Also, disable quantization if you're on a non-CUDA system.
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        except Exception as e:
            if ("gated repo" in str(e)) or ("Unauthorized" in str(e)) or isinstance(e, GatedRepoError):
                print("Authentication error detected. Attempting to re-login to Hugging Face...")
                hf_token = config["HF_KEY"]
                if not hf_token:
                    raise RuntimeError("HF_KEY environment variable is not set. Please set it with your Hugging Face token.")
                login(token=hf_token)
                # Retry after logging in
                tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=file_name, legacy=True)
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
            else:
                raise e

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            **kwargs
        )

    def get_pipeline(self, model_id: str = "bartowski/Mistral-Small-24B-Instruct-2501-GGUF", file_name: str = "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf", **kwargs):
        if self._pipeline is None:
            self._pipeline = self._create_pipeline(model_id, file_name, **kwargs)
        return self._pipeline
