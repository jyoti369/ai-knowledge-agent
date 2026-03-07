"""
Document Ingestion Pipeline.
Loads documents from the data/ directory, chunks them,
generates embeddings, and stores them in Pinecone.
"""

import sys
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import Config
from utils.document_loader import load_documents, chunk_documents
from utils.embeddings import get_embedding_model

console = Console()


def ensure_pinecone_index() -> None:
    """Create Pinecone index if it doesn't exist, or recreate if dimensions mismatch."""
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if Config.PINECONE_INDEX_NAME in existing_indexes:
        # Check if dimensions match
        index_info = pc.describe_index(Config.PINECONE_INDEX_NAME)
        current_dim = index_info.dimension
        if current_dim != Config.EMBEDDING_DIMENSION:
            console.print(
                f"[yellow]⚠️  Index '{Config.PINECONE_INDEX_NAME}' has dimension {current_dim}, "
                f"but we need {Config.EMBEDDING_DIMENSION}. Deleting and recreating...[/yellow]"
            )
            pc.delete_index(Config.PINECONE_INDEX_NAME)
            time.sleep(2)
        else:
            console.print(f"[green]✅ Pinecone index '{Config.PINECONE_INDEX_NAME}' already exists.[/green]")
            return

    console.print(f"[cyan]📦 Creating Pinecone index: {Config.PINECONE_INDEX_NAME} (dim={Config.EMBEDDING_DIMENSION})...[/cyan]")
    pc.create_index(
        name=Config.PINECONE_INDEX_NAME,
        dimension=Config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for index to be ready...", total=None)
        while not pc.describe_index(Config.PINECONE_INDEX_NAME).status.get("ready"):
            time.sleep(1)
        progress.update(task, description="Index is ready! ✅")


def ingest() -> None:
    """Run the full ingestion pipeline."""
    console.print(
        Panel(
            "[bold cyan]🚀 AI Knowledge Agent — Document Ingestion[/bold cyan]\n"
            "Loading, chunking, embedding, and storing documents in Pinecone.\n"
            "[dim]Using FREE models: HuggingFace embeddings + Google Gemini[/dim]",
            border_style="cyan",
        )
    )

    # Validate config (only Pinecone key needed for ingestion)
    if not Config.PINECONE_API_KEY:
        console.print("[red]❌ PINECONE_API_KEY is not set in .env[/red]")
        sys.exit(1)

    # Step 1: Load documents
    console.print("\n[bold]📄 Step 1: Loading Documents[/bold]")
    documents = load_documents()

    if not documents:
        console.print("[red]❌ No documents to ingest. Add files to the data/ directory.[/red]")
        sys.exit(1)

    console.print(f"   Loaded [green]{len(documents)}[/green] document(s).\n")

    # Step 2: Chunk documents
    console.print("[bold]✂️  Step 2: Chunking Documents[/bold]")
    chunks = chunk_documents(documents)
    console.print(
        f"   Split into [green]{len(chunks)}[/green] chunks "
        f"(size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}).\n"
    )

    # Step 3: Ensure Pinecone index exists
    console.print("[bold]🗄️  Step 3: Setting Up Pinecone Index[/bold]")
    ensure_pinecone_index()
    console.print()

    # Step 4: Generate embeddings (locally!) and store in Pinecone
    console.print("[bold]🔢 Step 4: Generating Embeddings (local) & Storing in Pinecone[/bold]")
    console.print(f"   Embedding model: [cyan]{Config.EMBEDDING_MODEL}[/cyan] (runs locally, free)")
    embeddings = get_embedding_model()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Embedding and upserting {len(chunks)} chunks...", total=None
        )
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=Config.PINECONE_INDEX_NAME,
            pinecone_api_key=Config.PINECONE_API_KEY,
        )
        progress.update(task, description="Done! ✅")

    console.print(
        Panel(
            f"[bold green]✅ Ingestion Complete![/bold green]\n\n"
            f"• Documents loaded: {len(documents)}\n"
            f"• Chunks created: {len(chunks)}\n"
            f"• Pinecone index: {Config.PINECONE_INDEX_NAME}\n"
            f"• Embedding model: {Config.EMBEDDING_MODEL} (free, local)\n\n"
            f"[dim]Run [bold]python agent.py[/bold] to start querying your knowledge base.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    ingest()
