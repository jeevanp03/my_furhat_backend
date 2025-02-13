from model_pipeline import ModelPipelineManager
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

class HuggingFaceLLM:
    def __init__(self, model_id: str = "bartowski/Mistral-Small-24B-Instruct-2501-GGUF", file_name: str="Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf", **kwargs):
        pipeline_instance = ModelPipelineManager().get_pipeline(model_id, file_name, **kwargs)
        self.chat_llm = self.__create_chat_pipeline(pipeline_instance)

    def __create_chat_pipeline(self, pipeline_instance):
        return ChatHuggingFace(HuggingFacePipeline(pipeline=pipeline_instance))
    
    def query(self, text: str):
        return self.chat_llm.invoke(text)


if __name__ == "__main__":
    llm = HuggingFaceLLM(max_new_tokens=150, top_k=40, temperature=0.2)
    print(llm.query("Hello, how are you?"))
