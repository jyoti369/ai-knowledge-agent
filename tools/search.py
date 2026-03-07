"""
Semantic search tool using Pinecone vector database.
"""

from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore

from config import Config
from utils.embeddings import get_embedding_model


@tool
def vector_search_tool(query: str) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Use this tool when you need to find specific information, facts, or details
    from the ingested documents. Returns the most relevant document chunks
    with their source information.

    Args:
        query: The search query describing what information you need.

    Returns:
        A formatted string containing the top matching document chunks
        with source citations and relevance scores.
    """
    embeddings = get_embedding_model()

    vectorstore = PineconeVectorStore(
        index_name=Config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=Config.PINECONE_API_KEY,
    )

    results = vectorstore.similarity_search_with_score(query, k=Config.TOP_K)

    if not results:
        return "No relevant documents found for this query."

    formatted_results = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source_file", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        relevance = f"{score:.2f}"

        formatted_results.append(
            f"--- Result {i} ---\n"
            f"Source: {source} (chunk {chunk_idx})\n"
            f"Relevance Score: {relevance}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n".join(formatted_results)
