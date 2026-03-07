"""
Multi-query retrieval tool.
Generates multiple query variations for better search recall.
"""

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

from config import Config
from utils.embeddings import get_embedding_model


@tool
def multi_query_search_tool(query: str) -> str:
    """
    Perform an enhanced search by generating multiple query variations.
    Use this tool when a single search query might miss relevant results,
    or when the user's question is complex and could be interpreted in
    multiple ways. This generates 3 alternative queries and combines
    the results for better coverage.

    Args:
        query: The original search query to expand and search with.

    Returns:
        Combined search results from multiple query variations,
        deduplicated and ranked by relevance.
    """
    # Generate query variations using LLM
    llm = ChatGroq(
        model=Config.LLM_MODEL,
        api_key=Config.GROQ_API_KEY,
        temperature=0.7,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful query expansion assistant. "
                "Given a user question, generate 3 alternative versions of the question "
                "that might help retrieve relevant information. "
                "Each variation should approach the question from a different angle. "
                "Return ONLY the 3 questions, one per line, numbered 1-3."
            )
        ),
        HumanMessage(content=f"Original question: {query}"),
    ]

    response = llm.invoke(messages)
    query_variations = [
        line.strip().lstrip("0123456789.)- ")
        for line in response.content.strip().split("\n")
        if line.strip()
    ][:3]

    # Search with all query variations
    all_queries = [query] + query_variations
    embeddings = get_embedding_model()

    vectorstore = PineconeVectorStore(
        index_name=Config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=Config.PINECONE_API_KEY,
    )

    # Collect unique results
    seen_content = set()
    unique_results = []

    for q in all_queries:
        results = vectorstore.similarity_search_with_score(q, k=3)
        for doc, score in results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score, q))

    # Sort by relevance score
    unique_results.sort(key=lambda x: x[1], reverse=True)

    if not unique_results:
        return "No relevant documents found across multiple query variations."

    formatted = [f"🔍 Searched with {len(all_queries)} query variations:\n"]
    for i, (doc, score, matched_query) in enumerate(unique_results[:Config.TOP_K], 1):
        source = doc.metadata.get("source_file", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        formatted.append(
            f"--- Result {i} ---\n"
            f"Source: {source} (chunk {chunk_idx})\n"
            f"Relevance: {score:.2f}\n"
            f"Matched Query: {matched_query}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n".join(formatted)
