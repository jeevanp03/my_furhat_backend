import os
from dotenv import load_dotenv, dotenv_values
from pathlib import Path


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
MOUNT_DIR = Path("/mnt")


config = dotenv_values(".env")

config.update({
    "HF_HOME": os.getenv("HF_HOME", str(MOUNT_DIR / "hf_cache")),
    "TORCH_HOME": os.getenv("TORCH_HOME", str(MOUNT_DIR / "torch_cache")),
    "VECTOR_STORE_PATH": os.getenv("VECTOR_STORE_PATH", str(MOUNT_DIR / "vector_store")),
    "DOCUMENTS_PATH": os.getenv("DOCUMENTS_PATH", str(MOUNT_DIR / "documents")),
    "MODEL_PATH": os.getenv("MODEL_PATH", str(MOUNT_DIR / "models")),
    "GGUF_MODELS_PATH": os.getenv("GGUF_MODELS_PATH", str(MOUNT_DIR / "models/gguf")),
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
})

# Set environment variables from config
for key, value in config.items():
    os.environ[key] = str(value)
