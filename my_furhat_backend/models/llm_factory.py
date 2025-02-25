from abc import ABC, abstractmethod
import multiprocessing
from my_furhat_backend.models.model_pipeline import ModelPipelineManager
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.chat_models import ChatLlamaCpp
from my_furhat_backend.config.settings import config
from my_furhat_backend.llm_tools.tools import tools as all_tools
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Abstract base class for LLMs
class BaseLLM(ABC):
    @abstractmethod
    def query(self, text: str, tool: bool = False):
        """Process a query, with optional tool integration."""
        pass

    @abstractmethod
    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        """Bind additional tools to the LLM."""
        pass

# Concrete implementation for HuggingFace
class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct", **kwargs):
        # self.chat_llm = self.__create_chat_pipeline(ModelPipelineManager().get_pipeline(model_id, **kwargs))
        kwargs.setdefault("max_new_tokens", 512)
        kwargs.setdefault("top_k", 50)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("repetition_penalty", 1.03)
        self.chat_llm = ChatHuggingFace(llm = self.__create_endpoint(model_id, **kwargs))
        # Uncomment if you want to pre-bind tools:
        # self.chat_llm_with_tools = self.chat_llm.bind_tools(all_tools)

    def __create_chat_pipeline(self, pipeline_instance):
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipeline_instance))
    
    def __create_endpoint(self, model_id: str, **kwargs):
        return HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            huggingfacehub_api_token=config["HF_KEY"],
            # callbacks=[StreamingStdOutCallbackHandler()],
            **kwargs
        )

    def query(self, text: str, tool: bool = False):
        if tool:
            return self.chat_llm_with_tools.invoke(text)
        else:
            return self.chat_llm.invoke(text)

    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        self.chat_llm.bind_tools(tools)

# Concrete implementation for Llama Cpp
class LlamaCcpLLM(BaseLLM):
    def __init__(self, model_id: str = "my_furhat_backend/ggufs_models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", **kwargs):
        kwargs.setdefault("n_ctx", 10000)
        kwargs.setdefault("n_gpu_layers", 18)
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("n_batch", 300)
        kwargs.setdefault("max_tokens", 450)
        kwargs.setdefault("repeat_penalty", 1.5)
        kwargs.setdefault("top_p", 0.5)
        kwargs.setdefault("verbose", True)
        self.chat_llm = ChatLlamaCpp(
            model_path=model_id,
            n_threads=multiprocessing.cpu_count() - 1,
            do_sample=True,
            **kwargs
        )
        # Uncomment if you want to pre-bind tools:
        # self.chat_llm_with_tools = self.chat_llm.bind_tools(all_tools)

    def query(self, text: str, tool: bool = False):
        if tool:
            return self.chat_llm_with_tools.invoke(text)
        else:
            return self.chat_llm.invoke(text)

    def bind_tools(self, tools: list, tool_schema: dict | str = None):
        self.chat_llm.bind_tools(tools)

# Factory method to create the appropriate LLM
def create_llm(llm_type: str, **kwargs) -> BaseLLM:
    if llm_type == "huggingface":
        return HuggingFaceLLM(**kwargs)
    elif llm_type == "llama":
        return LlamaCcpLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

# Export the functions and classes that should be publicly available
__all__ = ["create_llm", "HuggingFaceLLM", "LlamaCcpLLM", "BaseLLM"]
