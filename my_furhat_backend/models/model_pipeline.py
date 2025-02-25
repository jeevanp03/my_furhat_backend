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
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        **kwargs
    ):
        # Set default generation parameters
        kwargs.setdefault("max_new_tokens", 100)
        kwargs.setdefault("top_k", 50)
        kwargs.setdefault("temperature", 0.1)

        try:
            # Load tokenizer and model normally.
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            if ("gated repo" in str(e)) or ("Unauthorized" in str(e)) or isinstance(e, GatedRepoError):
                print("Authentication error detected. Re-logging into Hugging Face...")
                hf_token = config["HF_KEY"]
                if not hf_token:
                    raise RuntimeError("HF_KEY environment variable is not set.")
                login(token=hf_token)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                raise e

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            **kwargs
        )

    def get_pipeline(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3", **kwargs):
        if self._pipeline is None:
            self._pipeline = self._create_pipeline(model_id, **kwargs)
        return self._pipeline
