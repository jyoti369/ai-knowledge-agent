"""
Configuration module for AI Knowledge Agent.
Loads environment variables and provides centralized config access.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Groq (free tier — no credit card needed)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "knowledge-agent")

    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq") # 'groq' or 'ollama'
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11400")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "10"))

    # Paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Embedding dimensions for all-MiniLM-L6-v2
    EMBEDDING_DIMENSION: int = 384

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        errors = []
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set (required for Groq provider)")
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")

        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   • {error}")
            sys.exit(1)

# --- NETWORK FIX ---
# Force IPv4 for httpx/Groq client to prevent hanging on broken IPv6 networks
import socket
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfoIPv4(*args, **kwargs):
    responses = orig_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = getaddrinfoIPv4


