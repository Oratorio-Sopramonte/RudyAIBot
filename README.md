# ⛪ RudyAIBot: Smart Knowledge Assistant
An intelligent Telegram Bot powered by the Datapizza AI framework.

RudyAIBot is a RAG (Retrieval-Augmented Generation) system designed to provide instant, accurate answers about an Oratorio's regulations. By processing official PDF documentation, the bot acts as a 24/7 digital concierge for the community.

## 🚀 Key Features
- 📚 PDF-Grounded Intelligence: Uses a RAG pipeline to ensure all answers are derived strictly from official Oratorio documents.

- 💬 Telegram Integration: A seamless user interface accessible via any mobile device.

- 🧩 Datapizza AI Core: Built on the modular Datapizza framework 

- 🧠 ( TODO ) Session Memory: Maintains context for follow-up questions, allowing for natural, fluid conversations.

## 🛠 Tech Stack
- Framework: Datapizza AI
- Language: Python 3.10+
- Interface: python-telegram-bot
- LLM Support: Gemini API

## 📂 Project Structure
```Plaintext
oratorio-ai/
├── documents/           # Store your Oratorio PDFs here (Knowledge Base)
├── src/
│   ├── ingestion.py     # ETL Pipeline: Parses PDFs -> Embeddings -> Qdrant
│   ├── pipeline.py      # RAG Logic: Defines the Datapizza DagPipeline and LLM Client
│   └── bot.py           # Application Layer: Telegram Bot logic and message chunking
├── config/
│   └── ingestion_config.yaml # Configuration for chunking and paths
├── .env.example         # Template for API keys
└── requirements.txt     # Project dependencies
```

## ⚙️ Integration Architecture

The system relies on a modular **RAG (Retrieval-Augmented Generation)** architecture powered by **Datapizza AI**.

### 1. 📥 Ingestion Layer (`src/ingestion.py`)
This offline process converts raw PDF documents into a searchable vector index.
- **Parsing**: `DoclingParser` extracts text from PDFs, preserving structure.
- **Splitting**: `NodeSplitter` chunks text into manageable segments (512 tokens).
- **Embedding**: `FastEmbed` (Model: `BAAI/bge-small-en-v1.5`) converts text into dense vectors.
- **Storage**: **Qdrant** (Local Mode) stores the vectors on disk in `storage/qdrant_db`.

### 2. 🧠 Inference Pipeline (`src/pipeline.py`)
The `RAGService` class initializes a Directed Acyclic Graph (DAG) for processing user queries:
1.  **Query Embedding**: The user's question is embedded using the same BAAI model.
2.  **Vector Retrieval**: Qdrant performs a semantic search (Cosine Similarity) to find the top 5 most relevant context chunks.
3.  **Context Construction**: A `ChatPromptTemplate` combines the user query with the retrieved chunks.
4.  **LLM Generation**: **Google Gemini 2.5 Flash** generates an exhaustive answer based *strictly* on the provided context, following a refined system prompt.

### 3. 🤖 Application Layer (`src/bot.py`)
- **Interface**: Uses `python-telegram-bot` (Async).
- **Long Message Handling**: Automatically splits responses > 4096 characters to comply with Telegram API limits.
- **Admin Tools**: Includes `/update_kb` to trigger re-ingestion without restarting the bot.

## Setup
Step 1: Ingest Data
Run this once (or whenever PDFs change) to build the knowledge base.
```bash
python src/ingestion.py
```
Expected Output: "Index created and persisted to '.../storage'."

Step 2: Run the Bot
```bash
python src/bot.py
```