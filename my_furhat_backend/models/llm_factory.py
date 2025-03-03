from abc import ABC, abstractmethod
import multiprocessing
from my_furhat_backend.models.model_pipeline import ModelPipelineManager
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.chat_models import ChatLlamaCpp
from my_furhat_backend.config.settings import config
from my_furhat_backend.llm_tools.tools import tools as all_tools
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs):
        """
        Initialize the HuggingFaceLLM with a specified model and generation parameters.
        
        Parameters:
            model_id (str): The Hugging Face model identifier (default: "HuggingFaceTB/SmolLM2-1.7B-Instruct").
            **kwargs: Additional generation parameters for the model.
                     Defaults set include max_new_tokens, top_k, temperature, and repetition_penalty.
        """
        # Set default generation parameters if not provided
        kwargs.setdefault("max_new_tokens", 512)
        kwargs.setdefault("top_k", 50)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("repetition_penalty", 1.03)
        # Create a chat interface using the HuggingFaceEndpoint
        self.chat_llm = ChatHuggingFace(llm=self.__create_endpoint(model_id, **kwargs))
        # Uncomment the line below to pre-bind external tools if needed:
        # self.chat_llm_with_tools = self.chat_llm.bind_tools(all_tools)

    def __create_chat_pipeline(self, pipeline_instance):
        """
        Create a ChatHuggingFace pipeline from an existing pipeline instance.
        
        Parameters:
            pipeline_instance: An instance of a HuggingFace pipeline.
            
        Returns:
            ChatHuggingFace: A chat interface built on top of the given pipeline.
        """
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipeline_instance))
    
    def __create_endpoint(self, model_id: str, **kwargs):
        """
        Create a HuggingFaceEndpoint for text-generation with the provided model.
        
        Parameters:
            model_id (str): The model identifier to load.
            **kwargs: Additional parameters for the endpoint, including generation settings.
            
        Returns:
            HuggingFaceEndpoint: An endpoint configured for text-generation tasks.
        """
        return HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            huggingfacehub_api_token=config["HF_KEY"],
            # Uncomment the line below to enable streaming output callbacks:
            # callbacks=[StreamingStdOutCallbackHandler()],
            **kwargs
        )

    def query(self, text: str, tool: bool = False):
        """
        Process the query using the HuggingFace model.
        
        Parameters:
            text (str): The input query or prompt.
            tool (bool): If True, use the version of the model with pre-bound tools.
            
        Returns:
            The generated response as output from the chat model.
        """
        if tool:
            # If tools are enabled, use the tool-bound version (if pre-bound)
            return self.chat_llm_with_tools.invoke(text)
        else:
            # Otherwise, use the basic chat interface
            return self.chat_llm.invoke(text)

    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        """
        Bind external tools to the HuggingFaceLLM.
        
        Parameters:
            tools (list): A list of tools to be bound.
            tool_schema (dict or str, optional): The schema or configuration for these tools.
        """
        self.chat_llm.bind_tools(tools)

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
        kwargs.setdefault("n_gpu_layers", 12)
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
