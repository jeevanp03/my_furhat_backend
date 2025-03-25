from abc import ABC, abstractmethod
import multiprocessing
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.chat_models import ChatLlamaCpp
from my_furhat_backend.config.settings import config
from my_furhat_backend.llm_tools.tools import tools as all_tools
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import requests
from transformers import pipeline

# Abstract base class for all LLM implementations.
class BaseLLM(ABC):
    @abstractmethod
    def query(self, text: str, tool: bool = False):
        """
        Process a query with the language model.
        
        Parameters:
            text (str): The input text or prompt to be processed.
            tool (bool): If True, invoke the model with pre-bound tools.
            
        Returns:
            The generated response from the language model.
        """
        pass

    @abstractmethod
    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        """
        Bind external tools to the language model for extended functionality.
        
        Parameters:
            tools (list): A list of tools to be bound to the language model.
            tool_schema (dict or str, optional): The schema or configuration for the tools.
        """
        pass

# Concrete implementation of BaseLLM using HuggingFace models.
class HuggingFaceLLM(BaseLLM):
    """
    A class to interact with Hugging Face's API for language model inference.
    """
    def __init__(self, model_id: str, task: str = "text-generation", **kwargs):
        """
        Initialize the HuggingFaceLLM.

        Parameters:
            model_id (str): The ID of the model to use.
            task (str): The task type (default: "text-generation").
            **kwargs: Additional parameters for the model.
        """
        self.model_id = model_id
        self.task = task
        self.kwargs = kwargs
        
        # Set default generation parameters if not provided
        self.kwargs.setdefault("max_new_tokens", 512)
        self.kwargs.setdefault("temperature", 0.7)
        self.kwargs.setdefault("top_p", 0.9)
        self.kwargs.setdefault("do_sample", True)  # Enable sampling for temperature and top_p
        self.kwargs.setdefault("max_length", 1024)  # Set maximum sequence length
        
        # Initialize the API URL and headers
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        
        # Create the appropriate pipeline based on task
        self.llm = self.__create_pipeline()
    
    def __create_pipeline(self):
        """
        Create the appropriate pipeline based on the task type.
        """
        try:
            if self.task == "text-generation":
                return pipeline(
                    "text-generation",
                    model=self.model_id,
                    tokenizer=self.model_id,
                    device=-1,
                    **self.kwargs
                )
            elif self.task == "summarization":
                return pipeline(
                    "summarization",
                    model=self.model_id,
                    tokenizer=self.model_id,
                    device=-1,
                    **self.kwargs
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task}")
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return None
    
    def __truncate_input(self, prompt: str) -> str:
        """
        Truncate the input prompt if it exceeds the maximum token length.
        
        Parameters:
            prompt (str): The input prompt.
            
        Returns:
            str: The truncated prompt.
        """
        try:
            # Get the tokenizer from the pipeline
            tokenizer = self.llm.tokenizer if self.llm else None
            if not tokenizer:
                return prompt
                
            # Tokenize the input
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.kwargs["max_length"])
            
            # Decode back to text
            return tokenizer.decode(tokens["input_ids"][0])
        except Exception as e:
            print(f"Error truncating input: {e}")
            return prompt
    
    def query(self, prompt: str) -> str:
        """
        Query the model with the given prompt.

        Parameters:
            prompt (str): The input prompt.

        Returns:
            str: The model's response.
        """
        try:
            # Truncate the input if needed
            truncated_prompt = self.__truncate_input(prompt)
            
            if self.llm:
                # Use local pipeline
                result = self.llm(truncated_prompt, **self.kwargs)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", "")
                    return result[0]
                return str(result)
            else:
                # Fallback to API
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": truncated_prompt, **self.kwargs}
                )
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", "")
                    return result[0]
                return str(result)
        except Exception as e:
            print(f"Error in query: {e}")
            return ""

    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        """
        Bind tools to the LLM for enhanced functionality.
        """
        # Implementation for tool binding if needed
        pass

# Concrete implementation of BaseLLM using Llama Cpp.
class LlamaCcpLLM(BaseLLM):
    def __init__(self, model_id: str = "my_furhat_backend/ggufs_models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        """
        Initialize the LlamaCcpLLM with the specified model and generation parameters.
        
        Parameters:
            model_id (str): The path to the Llama Cpp model (default provided).
            **kwargs: Additional generation parameters for the Llama Cpp model.
                     Defaults include n_ctx, n_gpu_layers, temperature, n_batch, max_tokens,
                     repeat_penalty, top_p, and verbose.
        """
        # Set default parameters for Llama Cpp if not provided
        kwargs.setdefault("n_ctx", 10000)
        kwargs.setdefault("n_gpu_layers", 14)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("n_batch", 300)
        kwargs.setdefault("max_tokens", 512)
        kwargs.setdefault("repeat_penalty", 1.5)
        kwargs.setdefault("top_p", 0.5)
        kwargs.setdefault("verbose", True)
        # Create a ChatLlamaCpp instance with the specified model and system resources
        self.chat_llm = ChatLlamaCpp(
            model_path=model_id,
            n_threads=multiprocessing.cpu_count() - 1,
            do_sample=True,
            **kwargs
        )
        # Uncomment the line below to pre-bind external tools if needed:
        # self.chat_llm_with_tools = self.chat_llm.bind_tools(all_tools)

    def query(self, text: str, tool: bool = False):
        """
        Process the query using the Llama Cpp model.
        
        Parameters:
            text (str): The input query or prompt.
            tool (bool): If True, use the tool-bound version of the model.
            
        Returns:
            The generated response from the chat model.
        """
        if tool:
            # Use the tool-enhanced version if available
            return self.chat_llm_with_tools.invoke(text)
        else:
            # Otherwise, invoke the basic chat interface
            return self.chat_llm.invoke(text)

    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        """
        Bind external tools to the LlamaCcpLLM.
        
        Parameters:
            tools (list): A list of tools to be integrated.
            tool_schema (dict or str, optional): The schema or configuration for the tools.
        """
        self.chat_llm.bind_tools(tools)

def create_llm(llm_type: str, **kwargs) -> BaseLLM:
    """
    Factory function to create an instance of a language model based on the specified type.
    
    Parameters:
        llm_type (str): The type of LLM to create. Supported types are "huggingface" and "llama".
        **kwargs: Additional parameters to pass to the LLM constructor.
        
    Returns:
        BaseLLM: An instance of a concrete implementation of BaseLLM.
        
    Raises:
        ValueError: If the specified llm_type is not supported.
    """
    if llm_type == "huggingface":
        return HuggingFaceLLM(**kwargs)
    elif llm_type == "llama":
        return LlamaCcpLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

# Export the public classes and functions for external use.
__all__ = ["create_llm", "HuggingFaceLLM", "LlamaCcpLLM", "BaseLLM"]
