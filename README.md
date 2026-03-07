# 🤖 AI Knowledge Agent

A **Retrieval-Augmented Generation (RAG)** powered knowledge assistant that uses **Agentic AI** with **Pinecone** vector database to intelligently answer questions from your documents.

**✨ 100% Free Stack**: This version uses **Groq** for insanely fast LLM inference (Llama 3.3 70B) and **HuggingFace** for local embeddings, meaning zero subscription costs and no credit card required.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-Agents-green?logo=chainlink)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)
![Groq](https://img.shields.io/badge/Groq-Llama_3-orange)

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
│   Embeddings (Local: all-MiniLM-L6-v2)               │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- **Groq API Key** (Free, instant here: [console.groq.com/keys](https://console.groq.com/keys))
- **Pinecone API Key** (Free tier here: [app.pinecone.io](https://app.pinecone.io))

### 1. Installation

```bash
git clone https://github.com/jyoti369/ai-knowledge-agent.git
cd ai-knowledge-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
```

Edit your `.env` file and paste in your two API keys:

```bash
GROQ_API_KEY=gsk_your-groq-key
PINECONE_API_KEY=pcsk_your-pinecone-key
PINECONE_INDEX_NAME=knowledge-agent
```

---

## 💻 Commands & Expectations

### 1. Ingest Documents into the Database

Before asking questions, you need to give the agent knowledge. Simply place your files (`.pdf`, `.txt`, `.md`) into the `data/` folder and run:

```bash
python ingest.py
```

**What to expect:**
- The script automatically reads your files and splits them into 1000-character chunks.
- It generates embeddings (numerical representations) entirely locally on your machine using `all-MiniLM-L6-v2` (no internet upload required for embeddings).
- It creates the `knowledge-agent` index in Pinecone if it doesn't exist.
- It uploads the chunks to your Pinecone vector database.
- A neat terminal UI displays the progress.

### 2. Start the Interactive Agent

Once documents are ingested, start the interactive chat to query your documents:

```bash
python agent.py
```

**What to expect:**
- **Interactive Chat Interface**: You will enter a loop where you can ask back-to-back questions. Type `quit` to exit.
- **Autonomous Reasoning**: When you ask a question, you'll see a `🧠 Thinking...` message while the Agent (powered by Groq) decides what tools to use.
- **Tool Selection**:
  - `vector_search_tool`: Searches Pinecone for semantic matches to your query.
  - `summarize_tool`: Synthesizes lengthy retrieved chunks into a clean summary.
  - `multi_query_search_tool`: If the first search is bad, the agent re-words your question multiple times to cast a wider net.
- **Sourced Answers**: The agent will read the results and output a well-formatted answer, citing the specific document name (e.g., `Debojyoti_Mandal_Resume.pdf`) and chunk number it used to form the answer.

### 3. Run a Single Prompt (CLI Mode)

If you just want an answer and immediate exit (useful for scripts/automation):

```bash
python agent.py --query "Summarize the key experience points in my resume."
```

**What to expect:**
- Same behavior as above, but executes once and returns you directly back to your terminal shell.

---

## 🛠️ How it Works under the Hood

This project uses the **ReAct (Reasoning + Acting)** pattern via LangChain. Instead of a standard chatbot, it is an **Agent**:

1. **Thought** — "The user is asking about the resume. I need to search the database for 'resume experience'."
2. **Action** — Calls the `vector_search_tool`.
3. **Observation** — Reads the returned chunks from Pinecone.
4. **Repeat** — "These chunks are too long, I should summarize them" → Calls `summarize_tool`.
5. **Final Answer** — Delivers a comprehensive response with citations.

---

## 📝 License

MIT License — feel free to use this in your own projects!

*Built with ❤️ using LangChain, Groq, Pinecone, and HuggingFace Local Embeddings.*
