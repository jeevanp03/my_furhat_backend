"""
Language Model Factory Module

This module provides a factory pattern implementation for creating and managing different types of language models.
It supports both HuggingFace and LlamaCpp models with GPU optimization and monitoring capabilities.

Classes:
    BaseLLM: Abstract base class defining the interface for all LLM implementations.
    HuggingFaceLLM: Implementation using HuggingFace's API and models.
    LlamaCcpLLM: Implementation using LlamaCpp for local model inference.

Functions:
    create_llm: Factory function to create instances of different LLM types.
"""

from abc import ABC, abstractmethod
import multiprocessing
from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_models import ChatLlamaCpp
from my_furhat_backend.config.settings import config
from my_furhat_backend.utils.gpu_utils import setup_gpu, move_model_to_device, print_gpu_status, clear_gpu_cache
from transformers import pipeline
import torch
import os
import requests

class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""
    
    @abstractmethod
    def query(self, text: str, tool: bool = False) -> str:
        """
        Process a query with the language model.
        
        Args:
            text (str): The input text or prompt to be processed.
            tool (bool): If True, invoke the model with pre-bound tools.
            
        Returns:
            str: The generated response from the language model.
        """
        pass

    @abstractmethod
    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind external tools to the language model for extended functionality.
        
        Args:
            tools (list): A list of tools to be bound to the language model.
            tool_schema (dict | str, optional): The schema or configuration for the tools.
        """
        pass

class HuggingFaceLLM(BaseLLM):
    """
    Implementation of BaseLLM using HuggingFace's API and models.
    
    This class provides functionality to interact with HuggingFace models either through
    their API or local pipeline, with GPU optimization and monitoring capabilities.
    """
    
    def __init__(self, model_id: str, task: str = "text-generation", **kwargs):
        """
        Initialize the HuggingFaceLLM.

        Args:
            model_id (str): The ID of the model to use.
            task (str): The task type (default: "text-generation").
            **kwargs: Additional parameters for the model.
        """
        # Set up GPU and get device info
        device_info = setup_gpu()
        print_gpu_status()  # Print initial GPU status
        
        self.model_id = model_id
        self.task = task
        self.kwargs = kwargs
        
        # Set default generation parameters if not provided
        self.kwargs.setdefault("max_new_tokens", 512)
        self.kwargs.setdefault("temperature", 0.7)
        self.kwargs.setdefault("top_p", 0.9)
        self.kwargs.setdefault("do_sample", True)
        self.kwargs.setdefault("max_length", 1024)
        
        # Optimize for GPU if available
        if device_info["cuda_available"]:
            self.kwargs["device"] = 0
            self.kwargs["torch_dtype"] = torch.float16
            self.kwargs["low_cpu_mem_usage"] = True
        
        # Initialize the API URL and headers
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        
        # Create the appropriate pipeline based on task
        self.llm = self.__create_pipeline()
        
        # Move model to appropriate device if possible
        if self.llm:
            self.llm = move_model_to_device(self.llm, device_info["device"])
        
        # Print final GPU status after initialization
        print_gpu_status()
    
    def __del__(self):
        """Cleanup when the model is destroyed."""
        clear_gpu_cache()
    
    def __create_pipeline(self):
        """
        Create the appropriate pipeline based on the task type.
        
        Returns:
            Pipeline: The created pipeline or None if creation fails.
        """
        try:
            if self.task == "text-generation":
                return pipeline(
                    "text-generation",
                    model=self.model_id,
                    tokenizer=self.model_id,
                    device=0 if torch.cuda.is_available() else -1,
                    **self.kwargs
                )
            elif self.task == "summarization":
                return pipeline(
                    "summarization",
                    model=self.model_id,
                    tokenizer=self.model_id,
                    device=0 if torch.cuda.is_available() else -1,
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
        
        Args:
            prompt (str): The input prompt.
            
        Returns:
            str: The truncated prompt.
        """
        try:
            tokenizer = self.llm.tokenizer if self.llm else None
            if not tokenizer:
                return prompt
                
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.kwargs["max_length"])
            return tokenizer.decode(tokens["input_ids"][0])
        except Exception as e:
            print(f"Error truncating input: {e}")
            return prompt
    
    def query(self, prompt: str) -> str:
        """
        Query the model with the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The model's response.
        """
        try:
            clear_gpu_cache()
            print_gpu_status()
            
            truncated_prompt = self.__truncate_input(prompt)
            
            if self.llm:
                result = self.llm(truncated_prompt, **self.kwargs)
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        return result[0].get("generated_text", "")
                    return result[0]
                return str(result)
            else:
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
        finally:
            print_gpu_status()

    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind tools to the LLM for enhanced functionality.
        
        Args:
            tools (list): List of tools to bind.
            tool_schema (dict | str, optional): Schema for the tools.
        """
        pass

class LlamaCcpLLM(BaseLLM):
    """
    Implementation of BaseLLM using LlamaCpp for local model inference.
    
    This class provides functionality to interact with LlamaCpp models locally,
    with GPU optimization and monitoring capabilities.
    """
    
    def __init__(self, model_id: str = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        """
        Initialize the LlamaCcpLLM.
        
        Args:
            model_id (str): Name of the GGUF model file
            **kwargs: Additional generation parameters.
        """
        super().__init__()
        
        device_info = setup_gpu()
        print_gpu_status()
        
        # Get the full path to the GGUF model
        model_path = os.path.join(config["GGUF_MODELS_PATH"], model_id)
        
        # Verify model file exists and is accessible
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found at {model_path}")
            
        # Check file size to ensure it's not empty or corrupted
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Arbitrary minimum size for a GGUF model
            raise ValueError(f"Model file at {model_path} appears to be corrupted or incomplete (size: {file_size} bytes)")
            
        # Check file permissions
        if not os.access(model_path, os.R_OK):
            raise PermissionError(f"No read permission for model file at {model_path}")
            
        print(f"Loading model from: {model_path}")
        print(f"Model file size: {file_size / (1024*1024):.2f} MB")
        
        # Set default parameters
        kwargs.setdefault("n_ctx", 10000)
        kwargs.setdefault("n_gpu_layers", 32)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("n_batch", 512)
        kwargs.setdefault("max_tokens", 700)
        kwargs.setdefault("repeat_penalty", 1.5)
        kwargs.setdefault("top_p", 0.5)
        kwargs.setdefault("verbose", True)
        
        # Set n_threads only if not already provided in kwargs
        if "n_threads" not in kwargs:
            kwargs["n_threads"] = multiprocessing.cpu_count() - 1
        
        try:
            self.chat_llm = ChatLlamaCpp(
                model_path=model_path,
                do_sample=True,
                **kwargs
            )
            
            if device_info["cuda_available"]:
                self.chat_llm = move_model_to_device(self.chat_llm, device_info["device"])
            
            print_gpu_status()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Model path: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")
            print(f"Model file permissions: {oct(os.stat(model_path).st_mode)[-3:] if os.path.exists(model_path) else 'N/A'}")
            raise
    
    def __del__(self):
        """Cleanup when the model is destroyed."""
        clear_gpu_cache()

    def query(self, text: str, tool: bool = False) -> str:
        """
        Process the query using the LlamaCpp model.
        
        Args:
            text (str): The input query or prompt.
            tool (bool): If True, use the tool-bound version of the model.
            
        Returns:
            str: The generated response from the chat model.
        """
        print_gpu_status()
        
        try:
            clear_gpu_cache()
            
            if tool:
                response = self.chat_llm_with_tools.invoke(text)
            else:
                response = self.chat_llm.invoke(text)
            
            print_gpu_status()
            return response
        except Exception as e:
            print(f"Error in query: {e}")
            return ""

    def bind_tools(self, tools: list, tool_schema: dict | str = None) -> None:
        """
        Bind external tools to the LlamaCcpLLM.
        
        Args:
            tools (list): List of tools to bind.
            tool_schema (dict | str, optional): Schema for the tools.
        """
        self.chat_llm.bind_tools(tools)

def create_llm(llm_type: str, **kwargs) -> BaseLLM:
    """
    Factory function to create an instance of a language model.
    
    Args:
        llm_type (str): Type of LLM to create ("huggingface" or "llama").
        **kwargs: Additional parameters for the LLM constructor.
        
    Returns:
        BaseLLM: An instance of the specified LLM type.
        
    Raises:
        ValueError: If the specified llm_type is not supported.
    """
    if llm_type == "huggingface":
        return HuggingFaceLLM(**kwargs)
    elif llm_type == "llama":
        return LlamaCcpLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

__all__ = ["create_llm", "HuggingFaceLLM", "LlamaCcpLLM", "BaseLLM"]
