from my_furhat_backend.models.llm_factory import HuggingFaceLLM, LlamaCcpLLM
from my_furhat_backend.models.classifier import TextClassifier
from my_furhat_backend.llm_tools.tools import tools as all_tools
from my_furhat_backend.utils.util import clean_hc_response, format_chatml, format_structured_prompt, parse_structured_response
from pydantic import BaseModel, Field
import json


class Concierge:
    def __init__(
        self,
        model_instance=None,
        model_type: str = "hf",  # or "cpp"
        model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_path: str = "my_furhat_backend/ggufs_models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
        classifier_model_id: str = "facebook/bart-large-mnli",
        classifier_model_instance=None,
        **kwargs
    ):
        if model_instance is not None:
            self.llm = model_instance
        elif model_type == "hf":
            self.llm = HuggingFaceLLM(model_id=model_id, **kwargs)
        elif model_type == "cpp":
            self.llm = LlamaCcpLLM(model_id=model_path, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        if classifier_model_instance is not None:
            self.classifier = classifier_model_instance
        else:
            self.classifier = TextClassifier(model_id=classifier_model_id)
        
    def classify_query(self, query: str) -> str:
        """
        Classify the query into a category.
        """
        system_prompt = (
        "You are an expert query classifier. Classify the following query into one of these categories: "
        "API_HELP (query requires external API calls), NARROW_DOWN (query is ambiguous and needs clarification), or GENERAL (simple conversation that doesn't need further processing). "
        "Only output one of these labels."
        )
        full_prompt = f"{system_prompt}\nQuery: {query}\nCategory:"
        category = self.llm.query(full_prompt)
        return category.strip()

    def find_category(self, category: str) -> str:
        """
        Find places in a given category.
        """
        
        
        