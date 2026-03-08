"""
Summarization tool using Groq (free tier).
"""

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config import Config


@tool
def summarize_tool(text: str) -> str:
    """
    Summarize a piece of text or multiple document chunks into a concise summary.
    Use this tool when you have retrieved multiple document chunks and need to
    create a coherent, condensed summary of the information.

    Args:
        text: The text content to summarize. Can be raw document content
              or multiple concatenated search results.

    Returns:
        A concise, well-structured summary of the input text.
    """
    if Config.LLM_PROVIDER == "ollama":
        llm = ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.3,
        )
    else:
        llm = ChatGroq(
            model=Config.LLM_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=0.3,
            max_retries=0,
        )

    messages = [
        SystemMessage(
            content=(
                "You are a precise summarization assistant. "
                "Create a clear, concise summary of the provided text. "
                "Preserve key facts, figures, and important details. "
                "Use bullet points for multiple distinct pieces of information. "
                "Keep the summary under 200 words."
            )
        ),
        HumanMessage(content=f"Please summarize the following text:\n\n{text}"),
    ]

    response = llm.invoke(messages)
    return response.content
