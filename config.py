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
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Embedding dimensions for all-MiniLM-L6-v2
    EMBEDDING_DIMENSION: int = 384

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        errors = []
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set")
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY is not set")

        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   • {error}")
            print("\n💡 Copy .env.example to .env and fill in your API keys:")
            print("   cp .env.example .env")
            sys.exit(1)
