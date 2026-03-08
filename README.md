# 🤖 AI Knowledge Agent

A **Retrieval-Augmented Generation (RAG)** powered knowledge assistant that uses **Pinecone** vector database and local/cloud LLMs to accurately answer questions from your documents.

**✨ 100% Free Stack**: Uses **Ollama** for local LLM inference (zero rate limits, zero cost) and **HuggingFace** for local embeddings. No subscriptions, no credit cards, no API quotas.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Query                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│            Simple RAG Pipeline                       │
│                                                      │
│   1. Embed query (local HuggingFace model)           │
│   2. Search Pinecone for top-K relevant chunks       │
│   3. Send query + context to LLM                     │
│   4. Return accurate, cited answer                   │
│                                                      │
│   LLM Providers:                                     │
│   ┌──────────────┐  ┌──────────────┐                │
│   │  🦙 Ollama   │  │  ⚡ Groq     │                │
│   │  (Local, No  │  │  (Cloud,     │                │
│   │   Limits)    │  │   Fast)      │                │
│   └──────────────┘  └──────────────┘                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                 Pinecone Vector DB                    │
│                                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
│  │Chunk1│ │Chunk2│ │Chunk3│ │Chunk4│ │Chunk5│ ...   │
│  │ 🔢   │ │ 🔢   │ │ 🔢   │ │ 🔢   │ │ 🔢   │      │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │
│   Embeddings (Local: all-MiniLM-L6-v2)               │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- **Ollama** (Recommended — free, local, no rate limits): [ollama.com](https://ollama.com)
- **Pinecone API Key** (Free tier: [app.pinecone.io](https://app.pinecone.io))
- **Groq API Key** (Optional cloud alternative: [console.groq.com/keys](https://console.groq.com/keys))

### 1. Installation

```bash
git clone https://github.com/jyoti369/ai-knowledge-agent.git
cd ai-knowledge-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama & Pull a Model

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
```

### 3. Configuration

```bash
cp .env.example .env
```

Edit your `.env` file:

```bash
# Required
PINECONE_API_KEY=pcsk_your-pinecone-key
PINECONE_INDEX_NAME=knowledge-agent

# LLM Provider: 'ollama' (local, recommended) or 'groq' (cloud)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Only needed if LLM_PROVIDER=groq
# GROQ_API_KEY=gsk_your-groq-key
```

---

## 💻 Commands & Usage

### 1. Ingest Documents

Place your files (`.pdf`, `.txt`, `.md`) into the `data/` folder and run:

```bash
python ingest.py
```

**What happens:**
- Reads your files and splits them into 1000-character chunks with 200-character overlap.
- Generates embeddings locally using `all-MiniLM-L6-v2` (no data leaves your machine).
- Creates and populates the Pinecone index.

### 2. Interactive Chat

```bash
python agent.py
```

**What to expect:**
- Clean interactive chat interface.
- Instant responses to greetings (no LLM call needed).
- Accurate, sourced answers from your documents with citations.
- Type `quit` to exit, `help` for tips.

### 3. Single Query (CLI Mode)

```bash
python agent.py --query "What did Debojyoti contribute at Red Hat?"
```

Executes once and exits — useful for scripts and automation.

---

## 🛠️ How it Works

This project uses a **simple, reliable RAG (Retrieval-Augmented Generation) pipeline**:

1. **Embed** — Your query is converted to a vector using a local HuggingFace model.
2. **Retrieve** — Pinecone finds the top-K most semantically similar document chunks.
3. **Generate** — The query + retrieved context is sent to the LLM (Ollama or Groq).
4. **Answer** — The LLM produces a detailed answer citing exact document sources.

### Why Simple RAG over Agentic ReAct?

We intentionally chose a direct RAG pipeline over the more complex ReAct agent pattern because:

- **Reliability**: Small local models (8B params) struggle with ReAct's strict formatting requirements, causing parsing errors and timeouts.
- **Speed**: 1 LLM call instead of 3-5 iterative reasoning loops.
- **Accuracy**: The LLM focuses entirely on answering from context, not on tool-selection logic.

---

## ⚙️ Configuration Options

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | LLM backend: `ollama` (local) or `groq` (cloud) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `10` | Number of chunks retrieved per query |

---

## 📂 Project Structure

```
ai-knowledge-agent/
├── agent.py              # Main entry point (interactive + CLI)
├── ingest.py             # Document ingestion pipeline
├── config.py             # Centralized configuration
├── requirements.txt      # Python dependencies
├── data/                 # Place your documents here
│   ├── *.pdf
│   ├── *.txt
│   └── *.md
├── tools/                # Search, summarizer, multi-query tools
│   ├── search.py
│   ├── summarizer.py
│   └── multi_query.py
└── utils/                # Embedding model & document loader
    ├── embeddings.py
    └── document_loader.py
```

---

## 📝 License

MIT License — feel free to use this in your own projects!

*Built with ❤️ using LangChain, Ollama, Pinecone, and HuggingFace Local Embeddings.*
