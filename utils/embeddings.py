import logging
import warnings
import os
import sys

# Suppress annoying HuggingFace and sentence-transformers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Permanently kill the "unauthenticated requests" warning
# This prints via warnings.warn inside huggingface_hub — we intercept it globally
_original_warn = warnings.warn
def _silent_warn(message, *args, **kwargs):
    msg_str = str(message)
    if "unauthenticated" in msg_str.lower() or "HF_TOKEN" in msg_str:
        return  # swallow it
    _original_warn(message, *args, **kwargs)
warnings.warn = _silent_warn

try:
    import huggingface_hub.utils._http
    huggingface_hub.utils._http.DISABLE_TELEMETRY = True
except Exception:
    pass

from config import Config
from typing import Any


class SuppressOutput:
    """Aggressively suppress all stdout and stderr (perfect for stopping HF loading bars)."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

_embeddings_instance = None


def get_embedding_model() -> Any:
    """
    Get the configured HuggingFace embedding model.
    Runs locally — no API key or cost required.
    Uses caching so it doesn't reload on every tool call.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        with SuppressOutput():
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embeddings_instance

