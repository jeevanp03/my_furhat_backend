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

# Import the factory method from your LLM factory module.
from .llm_factory import create_llm

# Abstract base class for Chatbots.
class BaseChatbot(ABC):
    @abstractmethod
    def chatbot(self, state: dict) -> dict:
        """
        Process the conversation state and return an updated state.

        Parameters:
            state (dict): A dictionary containing the current conversation state,
                          typically including a list of messages.

        Returns:
            dict: The updated conversation state after processing the prompt.
        """
        pass

# Concrete implementation for HuggingFace-based chatbot.
class Chatbot_HuggingFace(BaseChatbot):
    def __init__(self, model_instance=None, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs):
        """
        Initialize the HuggingFace chatbot.

        If a model_instance is provided, it is used directly; otherwise, a new LLM instance is created
        using the factory method with the specified model_id and additional kwargs.

        Parameters:
            model_instance: An optional pre-initialized language model instance.
            model_id (str): The Hugging Face model identifier (default: "HuggingFaceTB/SmolLM2-1.7B-Instruct").
            **kwargs: Additional keyword arguments for model configuration.
        """
        if model_instance is not None:
            self.llm = model_instance
        else:
            # Create a HuggingFace LLM instance using the factory method.
            self.llm = create_llm("huggingface", model_id=model_id, **kwargs)
    
    def chatbot(self, state: dict) -> dict:
        """
        Process the conversation state using a HuggingFace model and update it with a new AI message.

        The method formats the conversation into a chat prompt, sends it to the language model,
        cleans the response, and appends it to the message history.

        Parameters:
            state (dict): The conversation state containing messages (list).

        Returns:
            dict: The updated state with the new AI-generated message appended.
        """
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")
        
        # Format the conversation messages into a prompt using a chatML format.
        prompt = format_chatml(messages)
        # Query the language model; tool integration can be enabled if desired.
        response = self.llm.query(prompt)
        
        # Process response based on its type.
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        
        # Clean the response text to remove any undesired formatting or artifacts.
        response_text = clean_hc_response(response_text)
        # Append the AI response to the conversation.
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state

# Concrete implementation for LlamaCpp-based chatbot.
class Chatbot_LlamaCpp(BaseChatbot):
    def __init__(self, model_instance=None, model_id: str = "my_furhat_backend/ggufs_models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        """
        Initialize the LlamaCpp chatbot.

        If a model_instance is provided, it is used directly; otherwise, a new LLM instance is created
        using the factory method with the specified model_id and additional kwargs.

        Parameters:
            model_instance: An optional pre-initialized language model instance.
            model_id (str): The model path or identifier for LlamaCpp (default provided).
            **kwargs: Additional keyword arguments for model configuration.
        """
        if model_instance is not None:
            self.llm = model_instance
        else:
            # Create a Llama LLM instance using the factory method.
            self.llm = create_llm("llama", model_id=model_id, **kwargs)
        
    def chatbot(self, state: dict) -> dict:
        """
        Process the conversation state using a LlamaCpp model and update it with a new AI message.

        This method formats the conversation using a structured prompt format specific to Llama,
        sends the prompt to the model, parses the structured response, and appends it to the message history.

        Parameters:
            state (dict): The conversation state containing messages (list).

        Returns:
            dict: The updated state with the new AI-generated message appended.
        """
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("No messages found in state.")
        
        # Format the conversation messages into a structured prompt.
        prompt = format_structured_prompt(messages)
        response = self.llm.query(prompt)
        
        # Process response based on its type.
        if isinstance(response, AIMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = str(response)
        
        # Parse the structured response to extract the desired content.
        response_text = parse_structured_response(response_text)
        # Append the parsed AI response to the conversation.
        messages.append(AIMessage(content=response_text))
        state["messages"] = messages
        return state

def create_chatbot(chatbot_type: str, **kwargs) -> BaseChatbot:
    """
    Factory function to create a chatbot instance based on the specified type.

    Parameters:
        chatbot_type (str): The type of chatbot to create. Supported types include "huggingface" and "llama".
        **kwargs: Additional keyword arguments for chatbot initialization.

    Returns:
        BaseChatbot: An instance of a concrete chatbot implementation.

    Raises:
        ValueError: If an unsupported chatbot type is provided.
    """
    if chatbot_type.lower() == "huggingface":
        return Chatbot_HuggingFace(**kwargs)
    elif chatbot_type.lower() == "llama":
        return Chatbot_LlamaCpp(**kwargs)
    else:
        raise ValueError(f"Unsupported chatbot type: {chatbot_type}")

# Export the public API for this module.
__all__ = ["create_chatbot", "Chatbot_HuggingFace", "Chatbot_LlamaCpp", "BaseChatbot"]
