# ⛪ RudyAIBot: Smart Knowledge Assistant
An intelligent Telegram Bot powered by the Datapizza AI framework.

RudyAIBot is a RAG (Retrieval-Augmented Generation) system designed to provide instant, accurate answers about an Oratorio's regulations. By processing official PDF documentation, the bot acts as a 24/7 digital concierge for the community.

## 🚀 Key Features
- 📚 PDF-Grounded Intelligence: Uses a RAG pipeline to ensure all answers are derived strictly from official Oratorio documents.

- 💬 Telegram Integration: A seamless user interface accessible via any mobile device.

- 🧩 Datapizza AI Core: Built on the modular Datapizza framework for advanced semantic search and agentic reasoning.

- 🔍 Source Citations: Every answer includes references to the specific sections/pages of the PDF source.

- 🧠 Session Memory: Maintains context for follow-up questions, allowing for natural, fluid conversations.

- ⚡ High Performance: Utilizes Qdrant for vector storage and semantic retrieval.

## 🛠 Tech Stack
- Framework: Datapizza AI

- Language: Python 3.10+

- Interface: python-telegram-bot

- Vector Database: Qdrant (Local or Cloud)

- LLM Support: OpenAI (GPT-4o), Anthropic (Claude), or local models via Ollama.

## 📂 Project Structure
```Plaintext
oratorio-ai/
├── data/                # Store your Oratorio PDFs here
├── src/
│   ├── ingestion.py     # Data parsing and vector indexing
│   ├── pipeline.py      # Datapizza RAG logic configuration
│   └── bot.py           # Telegram bot handlers and lifecycle
├── .env.example         # Template for API keys
└── requirements.txt     # Project dependencies
```

## ⚙️ Quick Start
Clone the repo:
```Bash
git clone https://github.com/yourusername/oratorio-ai.git
```
Install dependencies:
``` Bash
pip install -r requirements.txt
```
Configure Environment: Create a .env file with your TELEGRAM_TOKEN and GEMINI_API_KEY.

Index the PDF:
``` Bash
python src/ingestion.py
```

Run the Bot:
Bash
```
python src/bot.py
```