"""
AI Knowledge Agent — Main Entry Point.
Uses a simple RAG (Retrieval-Augmented Generation) pipeline to answer
questions from documents stored in Pinecone.
"""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import argparse
import sys
import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from config import Config
from utils.embeddings import get_embedding_model

console = Console()

# Simple greetings that don't need document lookup
GREETING_WORDS = {"hi", "hello", "hey", "hola", "greetings", "yo", "sup", "howdy"}

SYSTEM_PROMPT = """You are an intelligent knowledge assistant. You answer questions based ONLY on the provided context from documents.

Rules:
1. Answer ONLY using the provided context. Do not make up information.
2. Be specific — extract exact details, names, numbers, and bullet points from the context.
3. If someone asks about a specific company (e.g., "Red Hat"), ONLY include information that explicitly mentions that company. Do NOT mix in details from other companies.
4. Always cite the source document name at the end.
5. If the context doesn't contain relevant information, say "I couldn't find information about that in the documents."
6. Keep answers well-structured and detailed."""


def get_llm():
    """Create the LLM based on the configured provider."""
    if Config.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0,
        )
    else:
        return ChatGroq(
            model=Config.LLM_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=0,
            max_retries=0,
        )


def search_documents(query: str) -> str:
    """Search Pinecone for relevant document chunks."""
    embeddings = get_embedding_model()

    vectorstore = PineconeVectorStore(
        index_name=Config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=Config.PINECONE_API_KEY,
    )

    results = vectorstore.similarity_search_with_score(query, k=Config.TOP_K)

    if not results:
        return ""

    formatted = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source_file", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        formatted.append(
            f"[Document {i}: {source} (chunk {chunk_idx}, relevance: {score:.2f})]\n"
            f"{doc.page_content}\n"
        )

    return "\n".join(formatted)


def build_messages(query: str, context: str) -> list:
    """Build the LLM message list from query and retrieved context."""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Context from documents:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer the question based ONLY on the context above."
        )),
    ]


def stream_answer(query: str, llm, streaming: bool = True) -> str:
    """
    Run a single RAG query with streaming output.
    - If streaming=True: prints tokens live; returns full text at end.
    - If streaming=False: returns full text silently (used for --query CLI).
    """
    # Greetings: instant, no LLM call
    if query.strip().lower().rstrip("!?.") in GREETING_WORDS:
        msg = "Hello! I'm your AI Knowledge Agent. Ask me anything about your ingested documents. 👋"
        return msg

    # Step 1: Retrieve relevant documents from Pinecone
    context = search_documents(query)
    if not context.strip():
        return "I couldn't find any relevant information in the documents for your question."

    messages = build_messages(query, context)

    # Step 2: Stream tokens from LLM
    full_text = ""
    for chunk in llm.stream(messages):
        token = chunk.content
        full_text += token
        if streaming:
            console.print(token, end="", highlight=False, markup=False)

    if streaming:
        console.print()  # newline after streaming ends

    return full_text


def interactive_mode() -> None:
    """Run the agent in interactive chat mode."""
    Config.validate()
    llm = get_llm()

    provider = Config.LLM_PROVIDER.upper()
    model = Config.OLLAMA_MODEL if Config.LLM_PROVIDER == "ollama" else Config.LLM_MODEL

    console.print(
        Panel(
            f"[bold cyan]🤖 AI Knowledge Agent — Interactive Mode[/bold cyan]\n"
            f"[dim]Provider: {provider} | Model: {model}[/dim]\n"
            f"Ask questions about your ingested documents.\n"
            f"Type [bold]'quit'[/bold] to exit, [bold]'help'[/bold] for tips.",
            border_style="cyan",
        )
    )
    console.print()

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]👋 Goodbye![/dim]")
                break

            if query.lower() == "help":
                console.print(
                    Panel(
                        "💡 [bold]Tips:[/bold]\n"
                        "• Ask specific questions about your documents\n"
                        '• Try: "Who is Debojyoti?"\n'
                        '• Try: "What did Debojyoti do at Red Hat?"\n'
                        '• Try: "What are his technical skills?"\n'
                        "• Type 'quit' to exit",
                        border_style="yellow",
                    )
                )
                continue

            console.print("[dim]🧠 Thinking...[/dim]")

            # Greetings: print instantly in a panel
            if query.strip().lower().rstrip("!?.") in GREETING_WORDS:
                console.print(
                    Panel(
                        "Hello! I'm your AI Knowledge Agent. Ask me anything about your ingested documents. 👋",
                        title="[bold]🤖 Answer[/bold]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
                console.print()
                continue

            # Stream the answer with a header + live token output
            console.print()
            console.rule("[bold green]🤖 Answer[/bold green]", style="green")
            console.print()
            stream_answer(query, llm, streaming=True)
            console.print()
            console.rule(style="green")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]👋 Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\n")


def single_query_mode(query: str) -> None:
    """Run a single query and print the result."""
    Config.validate()
    llm = get_llm()

    console.print(f"[bold green]Query:[/bold green] {query}\n")
    console.print("[dim]🧠 Thinking...[/dim]")

    try:
        console.print()
        console.rule("[bold green]🤖 Answer[/bold green]", style="green")
        console.print()
        stream_answer(query, llm, streaming=True)
        console.print()
        console.rule(style="green")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Knowledge Agent — Ask questions about your documents"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query instead of interactive mode",
    )
    args = parser.parse_args()

    if args.query:
        single_query_mode(args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
