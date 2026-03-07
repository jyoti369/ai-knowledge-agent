"""
Document loading and chunking utilities.
Supports .txt, .md, and .pdf files.
"""

import os
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rich.console import Console

from config import Config

console = Console()


def load_documents(data_dir: str = None) -> List[Document]:
    """
    Load all supported documents from the data directory.

    Args:
        data_dir: Path to the directory containing documents.
                  Defaults to Config.DATA_DIR.

    Returns:
        List of loaded Document objects.
    """
    data_dir = data_dir or Config.DATA_DIR

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        console.print(f"[yellow]📁 Created data directory: {data_dir}[/yellow]")
        console.print("[yellow]   Place your documents (.txt, .pdf, .md) in this folder.[/yellow]")
        return []

    documents: List[Document] = []
    supported_extensions = {".txt", ".md", ".pdf"}

    files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not files:
        console.print("[yellow]⚠️  No supported documents found in data/ directory.[/yellow]")
        console.print("[yellow]   Supported formats: .txt, .md, .pdf[/yellow]")
        return []

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in (".txt", ".md"):
                loader = TextLoader(filepath, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(filepath)
            else:
                continue

            loaded_docs = loader.load()

            # Add source metadata
            for doc in loaded_docs:
                doc.metadata["source_file"] = filename

            documents.extend(loaded_docs)
            console.print(f"  ✅ Loaded: [green]{filename}[/green] ({len(loaded_docs)} page(s))")

        except Exception as e:
            console.print(f"  ❌ Error loading {filename}: [red]{e}[/red]")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks
