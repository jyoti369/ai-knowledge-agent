"""
Embedding generation utilities using OpenAI.
"""

from langchain_openai import OpenAIEmbeddings
from config import Config


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Get the configured OpenAI embedding model.

    Returns:
        OpenAIEmbeddings instance.
    """
    return OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
    )
