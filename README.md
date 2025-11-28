# HHN Intelligent Student Assistant

An intelligent campus assistant for students at Hochschule Heilbronn (HHN).  
It combines **graph-based routing** and **Retrieval-Augmented Generation (RAG)** on top of the **OpenAI API** and **Qdrant** to answer questions and provide step-by-step walking directions on campus.

---

## Features :

- **Smart Campus Routing**
  - Shortest path between buildings on TechCampus and BildungsCampus
  - Graph is built from real campus maps (nodes = entrances, edges = walkable paths)
  - Human-friendly, step-based English directions (no “north/south” or coordinates)

- **RAG-Based Question Answering**
  - Answers questions from official HHN PDFs, FAQs and web pages
  - Uses OpenAI embeddings + Qdrant vector search
  - Returns grounded answers with source citations

- **User-Friendly Web Interface**
  - Built with Streamlit
  - Chat-style UI
  - Shows both answers and their underlying sources

---

## Architecture Overview :

The system has two main pipelines:

1. **Routing Engine**
   - Campus maps are converted into a graph using:
     - `nodes.csv` (entrances, junctions, gates, etc.)
     - `edges.csv` / `edges_distances.csv` (walkable paths with distances)
   - Uses **NetworkX** to compute the shortest path between building entrances
   - A custom rephrase module converts raw paths into human-friendly route descriptions

2. **RAG Engine**
   - Data ingestion:
     - Extracts text from PDFs and selected HHN webpages
     - Cleans and chunks content into JSONL format
   - Embedding:
     - Uses **OpenAI `text-embedding-3-large`** to embed all chunks
   - Storage:
     - Stores embeddings and payloads in a **Qdrant** collection
   - Retrieval:
     - At query time, embeds the user question and retrieves top-k similar chunks
   - Generation:
     - Uses **OpenAI `gpt-4o-mini`** (or compatible) to generate a final answer grounded in retrieved context

The Streamlit UI decides whether a user message is:
- a **route request** → go to routing engine  
- or a **knowledge question** → go to RAG engine

---

## Tech Stack :

- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **LLM & Embeddings:** OpenAI API
- **Vector Database:** Qdrant (Cloud or Docker)
- **Graph Library:** NetworkX
- **PDF Processing:** PyPDF2
- **Other:** NumPy, Requests, python-dotenv

---

## Project Structure (simplified) :

```text
.
├── streamlit_app.py        # Main Streamlit entry point
├── requirements.txt
├── .env                    # Local environment variables (not committed)
├── data/
│   ├── raw/                # Original PDFs / HTML exports
│   ├── cleaned/            # Cleaned text / JSONL chunks
│   └── maps/
│       ├── nodes.csv       # Campus nodes (entrances, junctions, etc.)
│       └── edges_distances.csv  # Campus edges with distances
├── rag/
│   ├── ingest_qdrant.py    # Ingestion script: build embeddings + upload to Qdrant
│   ├── qdrant_store.py     # Qdrant client & retrieval utilities
│   └── runtime_qa.py       # High-level `answer_query()` function
└── router/
    ├── router.py           # Graph loading + shortest path routing
    └── router_utils.py     # Distance/time helpers & rephrase logic
