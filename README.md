# SoAgent – Social-to-Lead AI Agent 🤖

A Retrieval-Augmented Generation (RAG) system built with LangChain that answers questions about SocialFlow, a fictional SaaS platform for social media conversation automation. The system uses deterministic intent detection, vector search, and conditional tool chaining to provide accurate, context-aware responses.

## What It Does

SoAgent is an agentic RAG system that helps users learn about SocialFlow's features, pricing, supported platforms, and limitations. When users show business interest, it can also initiate lead capture conversations. The system combines keyword-based intent detection with semantic search to route queries to the right knowledge sources before generating answers.

## How It Works

The system follows a simple but effective flow:

1. **Intent Detection** – Keyword-based matching identifies what the user wants (overview, features, pricing, platforms, limitations, or business inquiries). No LLM needed here, just fast pattern matching.

2. **Context Retrieval** – For non-business intents, the system searches the FAISS vector database using semantic similarity. It filters results by intent and retrieves the top 2-3 relevant chunks from the knowledge base.

3. **Business Handling** – If the user shows business interest (buy, demo, trial) along with other intents, a separate Gemini-powered conversation tool engages them for lead capture.

4. **Answer Generation** – Gemini generates the final answer using the retrieved context. If no context is found, it answers generally based on its training.

The entire flow uses conditional if-else logic rather than complex agent frameworks. This makes the system predictable, debuggable, and easy to understand.

## Knowledge Base

The knowledge base contains five text files covering different aspects of SocialFlow:

- **overview.txt** – What SocialFlow is, its purpose, and target audience
- **features.txt** – 11 core features including automated replies, multi-platform support, intent detection, and lead capture
- **pricing.txt** – Four pricing tiers (Starter, Growth, Business, Enterprise) plus free trial information
- **supported_platforms.txt** – Platforms like Instagram, Facebook Messenger, LinkedIn, and website chat
- **limitation.txt** – Known limitations and constraints of the platform

Each file is split into chunks of 500 characters with 50-character overlap, embedded using `sentence-transformers/all-mpnet-base-v2`, and stored in a FAISS vector database for fast similarity search.

## Tech Stack

- **LangChain** – Framework for building the RAG pipeline and tool orchestration
- **FAISS** – Vector database for semantic search 🔍
- **sentence-transformers** – Local embedding model (all-mpnet-base-v2, 768 dimensions)
- **Gemini 2.5 Flash** – Google's LLM for answer generation and business conversations
- **Streamlit** – Simple web interface
- **Python 3.10+** – Core language

## Installation

1. Clone or navigate to the project directory.

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

Start the Streamlit app:
```bash
streamlit run RAG_Agent/main.py
```

On first run, the system will automatically ingest the knowledge base into FAISS. This takes 2-5 minutes as it downloads the embedding model and processes all documents. Subsequent runs are instant since the database persists.

## Design Philosophy

This project intentionally avoids complex agent frameworks and planners. Instead, it uses straightforward conditional logic to route queries. The philosophy is simple: **intent decides where to search, embeddings decide what is relevant, and LLMs only generate language**. This makes the system transparent, maintainable, and suitable for production use cases where predictability matters.

## Notes

- The embedding model runs locally and offline. No API calls needed for embeddings. ⚡
- Intent detection is deterministic and fast, using keyword matching rather than LLM classification.
- The FAISS database is created once and reused across sessions.
- Business lead capture only triggers when business intent appears alongside other intents. 💼

---

Built as a demonstration of agentic RAG systems with deterministic routing and semantic search.

