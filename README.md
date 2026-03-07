# 🤖 AI Knowledge Agent

A **Retrieval-Augmented Generation (RAG)** powered knowledge assistant that uses **Agentic AI** with **Pinecone** vector database to intelligently answer questions from your documents.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green?logo=chainlink)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange?logo=openai)

---

## 🧠 What is This?

This project demonstrates a practical use-case of **Agentic AI** combined with a **Vector Database (Pinecone)**:

1. **Ingest** — Load documents (`.txt`, `.pdf`, `.md`) → chunk them → generate embeddings → store in Pinecone
2. **Ask** — Query the AI agent with natural language questions
3. **Agent Reasons** — The agent autonomously decides which tools to use:
   - 🔍 **Semantic Search** — Find the most relevant document chunks
   - 📝 **Summarize** — Generate concise summaries of retrieved content
   - 🔗 **Multi-Query** — Reformulate your question for better retrieval
4. **Answer** — Returns a well-structured answer with source citations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Query                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              🤖 LangChain Agent (ReAct)              │
│                                                      │
│   Thinks → Selects Tool → Observes → Responds       │
│                                                      │
│   Tools:                                             │
│   ┌─────────────┐ ┌──────────┐ ┌─────────────────┐  │
│   │ Vector Search│ │Summarizer│ │ Multi-Query     │  │
│   │ (Pinecone)  │ │ (LLM)    │ │ Retriever       │  │
│   └──────┬──────┘ └────┬─────┘ └────────┬────────┘  │
│          │             │                │            │
└──────────┼─────────────┼────────────────┼────────────┘
           │             │                │
           ▼             ▼                ▼
┌──────────────────────────────────────────────────────┐
│                 Pinecone Vector DB                    │
│                                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│  │Chunk1│ │Chunk2│ │Chunk3│ │Chunk4│ │Chunk5│ ...   │
│  │ 🔢   │ │ 🔢   │ │ 🔢   │ │ 🔢   │ │ 🔢   │      │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │
│     Embeddings (OpenAI text-embedding-3-small)       │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API Key
- Pinecone API Key (free tier works!)

### 1. Clone the Repository

```bash
git clone https://github.com/jyoti369/ai-knowledge-agent.git
cd ai-knowledge-agent
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=sk-your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=knowledge-agent
```

### 5. Ingest Documents

Place your documents in the `data/` folder, then run:

```bash
python ingest.py
```

Supported formats: `.txt`, `.pdf`, `.md`

### 6. Ask Questions

**Interactive Mode (CLI Chat):**

```bash
python agent.py
```

**Single Query Mode:**

```bash
python agent.py --query "What are the key concepts discussed in the documents?"
```

---

## 📂 Project Structure

```
ai-knowledge-agent/
├── agent.py              # Main agentic AI entry point (ReAct agent)
├── ingest.py             # Document ingestion pipeline
├── config.py             # Configuration & environment variables
├── tools/
│   ├── __init__.py
│   ├── search.py         # Semantic search tool (Pinecone)
│   ├── summarizer.py     # Document summarization tool
│   └── multi_query.py    # Multi-query retrieval tool
├── utils/
│   ├── __init__.py
│   ├── document_loader.py# Load & chunk documents
│   └── embeddings.py     # Embedding generation utilities
├── data/                 # Place your documents here
│   └── sample.txt        # Sample document for testing
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 💡 Example Usage

```
🤖 AI Knowledge Agent - Interactive Mode
Type 'quit' to exit, 'clear' to reset conversation.
──────────────────────────────────────────

You: What is retrieval-augmented generation?

🧠 Agent is thinking...
   → Using tool: vector_search
   → Found 3 relevant chunks
   → Using tool: summarizer

🤖 Answer:
Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation. Instead of relying solely
on a language model's training data, RAG first searches a knowledge
base (in this case, Pinecone vector DB) for relevant information,
then uses that context to generate accurate, grounded answers.

📚 Sources:
  • sample.txt (chunk 2, relevance: 0.94)
  • sample.txt (chunk 5, relevance: 0.89)
```

---

## 🛠️ How the Agent Works

This project uses the **ReAct (Reasoning + Acting)** pattern via LangChain:

1. **Thought** — The agent reasons about what information it needs
2. **Action** — It selects and calls the appropriate tool
3. **Observation** — It processes the tool's output
4. **Repeat** — Until it has enough information to answer
5. **Final Answer** — Delivers a comprehensive response with citations

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `vector_search` | Performs semantic similarity search in Pinecone |
| `summarize_results` | Summarizes retrieved document chunks |
| `multi_query_search` | Generates multiple query variations for better recall |

---

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings & LLM | Required |
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index | `knowledge-agent` |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | OpenAI chat model | `gpt-4o-mini` |
| `CHUNK_SIZE` | Document chunk size (chars) | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K` | Number of results to retrieve | `5` |

---

## 📝 License

MIT License — feel free to use this in your own projects!

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with ❤️ using LangChain, Pinecone, and OpenAI*
