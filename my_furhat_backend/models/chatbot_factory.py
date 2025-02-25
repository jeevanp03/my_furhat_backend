import re
from abc import ABC, abstractmethod
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from my_furhat_backend.llm_tools.tools import tools as all_tools
from my_furhat_backend.utils.util import (
    clean_hc_response,
    format_chatml,
    format_structured_prompt,
    parse_structured_response,
)
from pydantic import BaseModel, Field
import json

# Import the factory method from your LLM factory
from .llm_factory import create_llm

# Abstract base class for Chatbots
class BaseChatbot(ABC):
    @abstractmethod
    def chatbot(self, state: dict) -> dict:
        """Process the conversation state and return an updated state."""
        pass

# Concrete implementation for HuggingFace-based chatbot
class Chatbot_HuggingFace(BaseChatbot):
    def __init__(self, model_instance=None, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs):
        if model_instance is not None:
            self.llm = model_instance
        else:
            # Use the factory method to create a HuggingFace LLM instance
            self.llm = create_llm("huggingface", model_id=model_id, **kwargs)
    
    def chatbot(self, state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")
        prompt = format_chatml(messages)
        # Pass tool=True to enable any tool integration
        response = self.llm.query(prompt)
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        response_text = clean_hc_response(response_text)
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state

# Concrete implementation for LlamaCpp-based chatbot
class Chatbot_LlamaCpp(BaseChatbot):
    def __init__(self, model_instance=None, model_id: str = "my_furhat_backend/ggufs_models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        if model_instance is not None:
            self.llm = model_instance
        else:
            # Use the factory method to create a Llama LLM instance
            self.llm = create_llm("llama", model_id=model_id, **kwargs)
        
    def chatbot(self, state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")
        # Use a structured prompt format for Llama
        prompt = format_structured_prompt(messages)
        response = self.llm.query(prompt)
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        response_text = parse_structured_response(response_text)
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state

# Factory method to create the appropriate chatbot instance
def create_chatbot(chatbot_type: str, **kwargs) -> BaseChatbot:
    if chatbot_type.lower() == "huggingface":
        return Chatbot_HuggingFace(**kwargs)
    elif chatbot_type.lower() == "llama":
        return Chatbot_LlamaCpp(**kwargs)
    else:
        raise ValueError(f"Unsupported chatbot type: {chatbot_type}")

# Export the public API for this module
__all__ = ["create_chatbot", "Chatbot_HuggingFace", "Chatbot_LlamaCpp", "BaseChatbot"]
