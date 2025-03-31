import os
from dotenv import load_dotenv, dotenv_values

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

config = dotenv_values(".env")

config.update({
    "TRANSFORMERS_CACHE": "/mnt/models/caches/huggingface",
    "TORCH_HOME": "/mnt/models/caches/torch",
    "VECTOR_STORE_PATH": "/mnt/data/vector_store",
    "DOCUMENTS_PATH": "/mnt/data/documents",
    "MODEL_PATH": "/mnt/models/weights",
    "CUDA_VISIBLE_DEVICES": "0"
})