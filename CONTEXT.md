## 📋 Full Feature List
**1. Knowledge Ingestion (Admin Side)**
- Multi-Source PDF Parsing: Support for reading complex PDF layouts (tables, headers, footnotes) using Datapizza's DoclingParser.
- Semantic Chunking: Intelligent text splitting that maintains context across sections of the Oratorio regulations.
- Vector Indexing: Automated embedding generation and storage in Qdrant (standard for Datapizza AI) for fast semantic search.

**2. Intelligent Retrieval (RAG)**
- Hybrid Search: Combined semantic and keyword search to find specific Oratorio rules accurately.
- Source Citations: The bot identifies which specific page or section of the PDF the answer was derived from.
- Reranking: Integration of a reranker (like Cohere) to prioritize the most relevant text snippets before generating the answer.

**3. Telegram Interface**
- Natural Language Interaction: Users can ask questions in plain Italian/English.
- Session Management: Persistent memory per user to handle follow-up questions (e.g., "And what about children under 10?").- Admin Commands: Dedicated commands (e.g., /update_kb) to trigger re-indexing of the PDF knowledge base.

**4. Observability & Maintenance**
- Traceability: Full logging of the RAG pipeline steps to debug "hallucinations.
- Model Agnosticism: Ability to swap the brain (OpenAI, Gemini, or local Llama 3) without changing the Telegram logic.


## 🔄 General System Workflow
The system operates in two distinct phases:

**Phase A: The Ingestion Pipeline (Static/Triggered)**
- Load: The IngestionPipeline monitors a local folder or receiving a command to read oratorio_rules.pdf.
- Parse & Split: The PDF is parsed into structured nodes.
- Embed & Store: Nodes are converted into vectors and stored in the Qdrant Vector DB.

**Phase B: The Chat Workflow (Dynamic)**
- Input: User sends a question via Telegram.Query Rewriting: The system clarifies the question (e.g., handling "What time?" by expanding to "What time does the Oratorio open?").
- Retrieval: The DagPipeline searches the Vector DB for the top $k$ relevant text chunks.
- Generation: The LLM receives the chunks + the question and generates a grounded response.
- Output: The bot sends the response to the user with a reference to the source.

## 🛠 Project Structure (Folder Tree)
```Plaintext
RudyAIBot/
├── data/                # Store your Oratorio PDFs here
├── src/
│   ├── ingestion.py     # Data parsing and vector indexing
│   ├── pipeline.py      # Datapizza RAG logic configuration
│   └── bot.py           # Telegram bot handlers and lifecycle
├── .env.example         # Template for API keys
└── requirements.txt     # Project dependencies
```
