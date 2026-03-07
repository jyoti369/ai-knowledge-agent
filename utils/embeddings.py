import logging
import warnings
import os

# Suppress annoying HuggingFace and sentence-transformers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings
from config import Config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Get the configured HuggingFace embedding model.
    Runs locally — no API key or cost required.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
