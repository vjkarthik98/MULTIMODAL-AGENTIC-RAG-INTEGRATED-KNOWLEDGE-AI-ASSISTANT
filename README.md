# 🧠 Multimodal RAG Knowledge Assistant

[![CI](https://github.com/vjkarthik98/multimodal-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vjkarthik98/multimodal-rag-assistant/actions/workflows/ci.yml)
[![Eval Gate](https://github.com/vjkarthik98/multimodal-rag-assistant/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/vjkarthik98/multimodal-rag-assistant/actions/workflows/eval-gate.yml)
[![CD](https://github.com/vjkarthik98/multimodal-rag-assistant/actions/workflows/cd.yml/badge.svg)](https://github.com/vjkarthik98/multimodal-rag-assistant/actions/workflows/cd.yml)

## 🚀 Overview

A production-grade **Multimodal Retrieval-Augmented Generation (RAG) system** that supports **text, PDFs, images, audio, and video**, with memory-aware and reasoning-enhanced responses.

> ⚡ Fully runnable locally with open-source models (no API dependency)
> 🎯 Startup-grade GenAI system with multimodal processing, memory, reasoning, and local LLM inference

---

## 🔥 Key Highlights

* 🧠 Multimodal ingestion (Text, PDF, Image, Audio, Video)
* 🔄 Unified pipeline (all modalities → text → embeddings)
* ✂️ Recursive chunking + metadata enrichment
* 🎯 Audio & Video query support (Speech-to-Text → RAG)
* 🧠 Memory-enabled RAG (Redis + MongoDB)
* 🧠 Memory summarization for long conversations
* 🧩 Reasoning layer for context + memory fusion
* ⚡ Local LLM inference (GGUF via llama.cpp)
* 🚀 FastAPI backend (modular architecture)
* 🎛️ Gradio UI (end-to-end interaction)

---

## ✨ Features

### 🔍 Core RAG

* Semantic retrieval using embeddings
* Context-aware answer generation
* Top-k chunk retrieval

### 🧠 Retrieval Optimization

* Recursive chunking (chunk_size=500, overlap=100)
* Chunk-level storage (not document-level)
* Metadata-aware retrieval

### 🎯 Multimodal Intelligence

* **PDF** → text extraction
* **Image** → BLIP captioning + OCR (Tesseract)
* **Audio** → Whisper (speech-to-text)
* **Video** → frame extraction + audio extraction → text

### 🧠 Memory System

* Redis → short-term conversational memory
* MongoDB → persistent chat history
* Memory summarization → compress long context

### 🧩 Reasoning Layer

* Combines retrieved context + memory
* Enhances prompts dynamically
* Improves response relevance

### 🌐 Full Stack System

* FastAPI backend
* Gradio UI frontend
* API + UI integration

---

## ❗ Problem Statement

Traditional systems:

* Fail to understand semantic meaning
* Lose context due to poor chunking
* Ignore multimodal data
* Lack conversational memory

---

## 💡 Solution

This system implements a **unified multimodal RAG pipeline**:

* All inputs → converted to text
* Smart chunking with overlap
* Embedding via MiniLM
* Storage in Qdrant (vector DB)
* Retrieval (top-k chunks)
* Memory injection (Redis + summaries)
* Reasoning layer (context fusion)
* LLM generation (GGUF via llama.cpp)

---

## 🧪 Installation & Setup

### 🔧 Prerequisites

* Python 3.10+
* Docker
* FFmpeg
* Tesseract OCR

---

### 📦 Clone Repository

```bash id="cl1"
git clone https://github.com/vjkarthik98/multimodal-rag-assistant.git
cd multimodal-rag-assistant
```

---

### 🧱 Environment Setup

```bash id="cl2"
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```

---

### 🐳 Run Required Services

```bash id="cl3"
# Qdrant (Vector DB)
docker run -p 6333:6333 qdrant/qdrant

# Redis (Short-term memory)
docker run -p 6379:6379 redis

# MongoDB (Persistent memory)
docker run -p 27017:27017 mongo
```

---

### 🧠 Setup GGUF Model

Place model file:

```id="cl4"
models/mistral/mistral-7b-instruct.Q4_K_M.gguf
```

---

### 🚀 Run Backend

```bash id="cl5"
uvicorn app.main:app --reload
```

---

### 🎛️ Run UI

```bash id="cl6"
python gradio_app.py
```

---

### 🌐 Access

* API Docs → http://127.0.0.1:8000/docs
* UI → http://127.0.0.1:7860

---

## 🌐 API Layer

### 🔍 Query Endpoint

```http id="cl7"
POST /rag/query
```

```json id="cl8"
{
  "query": "Explain artificial intelligence"
}
```

---

### 📁 Upload Endpoint

```http id="cl9"
POST /upload/file
```

Supports:

* Text / PDF / Image / Audio / Video

---

## 🧠 Memory Architecture

### Memory Types

* **Qdrant** → Knowledge memory (embeddings)
* **Redis** → Short-term working memory
* **MongoDB** → Persistent chat memory

### 🔄 Memory Flow

User Query
↓
Redis (recent context)
↓
Retriever (Qdrant knowledge)
↓
Reasoning Layer
↓
LLM
↓
Response
↓
MongoDB

---

## 🏗️ System Architecture

### 🔷 High-Level Flow

User Input
↓
Modality Detection
↓
Text Conversion (OCR / STT / Video Processing)
↓
Chunking + Metadata
↓
Embedding (MiniLM)
↓
Qdrant
↓
Retriever
↓
Memory (Redis + Summary)
↓
Reasoning Layer
↓
LLM (GGUF via llama.cpp)
↓
Response

---

## 🧱 Architecture Layers

1. **Frontend Layer**

   * Gradio UI

2. **API Layer**

   * FastAPI routes

3. **Pipeline Layer**

   * Orchestrates full RAG flow

4. **Processing Layer**

   * Ingestion + chunking + embeddings

5. **Retrieval Layer**

   * Qdrant vector search

6. **Memory Layer**

   * Redis + MongoDB

7. **Reasoning Layer**

   * Context + memory fusion

8. **LLM Layer**

   * GGUF inference (llama.cpp)

9. **Infrastructure Layer**

   * Docker + model offloading

---

## ⚙️ Tech Stack

### 🧠 AI / ML

* Sentence Transformers (MiniLM)
* Mistral 7B (GGUF via llama.cpp)
* BLIP (image captioning)
* Whisper (speech-to-text)

### ⚡ Backend

* FastAPI

### 🎛️ Frontend

* Gradio

### 🗄️ Data

* Qdrant

### 🧠 Memory

* Redis + MongoDB

### 🔄 Processing

* Tesseract OCR
* FFmpeg (video/audio processing)

---

## 📂 Project Structure

```text
project_root/
│
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── rag_routes.py          # FastAPI routes
│   │
│   ├── core/                          # Config, logging
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── ingestion/                     # Multimodal ingestion
│   │   ├── text_ingest.py
│   │   ├── document_ingest.py
│   │   ├── image_ingest.py
│   │   ├── audio_ingest.py
│   │   ├── video_ingest.py
│   │   ├── frame_captioner.py         # BLIP captioning
│   │   ├── video_frames.py
│   │   ├── router.py                  # Modality routing
│   │   ├── schema.py
│   │   └── pipeline.py
│   │
│   ├── embeddings/                    # Embedding logic
│   │   ├── text_embedder.py
│   │   ├── image_embedder.py
│   │   ├── audio_embedder.py
│   │   ├── video_embedder.py
│   │   └── clip_text_embedder.py
│   │
│   ├── vectorstore/                   # Qdrant integration
│   │   └── main.py
│   │
│   ├── retrieval/                     # Retrieval pipeline
│   │   ├── retriever.py
│   │   └── query_pipeline.py
│   │
│   ├── prompt/                        # Prompt construction
│   │   └── prompt_builder.py
│   │
│   ├── reasoning/                     # Reasoning layer
│   │   ├── query_decomposer.py
│   │   ├── reasoning_engine.py
│   │   └── result_fusion.py
│   │
│   ├── llm/                           # GGUF model interface
│   │   ├── gguf_model.py
│   │   └── mistral_loader.py
│   │
│   ├── memory/                        # Memory system
│   │   ├── memory_manager.py
│   │   ├── redis_memory.py
│   │   ├── mongo_memory.py
│   │   ├── memory_filter.py
│   │   ├── memory_fusion.py
│   │   ├── formatter.py
│   │   └── summarizer.py
│   │
│   ├── pipeline/                      # RAG orchestration
│   │   └── rag_pipeline.py
│   │
│   ├── utils/                         # Utilities
│   │   ├── chunking.py
│   │   └── logger.py
│   │
│   └── main.py                        # FastAPI entry point
│
├── configs/                           # Environment configs
├── data/
│   ├── raw/                           # Raw multimodal data
│   └── processed/                     # Processed data
│
├── docs/                              # Documentation & images
├── models/                            # GGUF models
├── notebooks/                         # Experiments
├── offload/                           # Model optimization / CPU offloading
│
├── scripts/                           # Utility scripts
│   ├── download_model.py
│   ├── ingest_data.py
│   ├── test_gguf.py
│   └── test_stream.py
│
├── tests/                             # Test suite
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_memory.py
│   ├── test_pipeline.py
│   └── ... (comprehensive module tests)
│
├── gradio_app.py                      # UI layer
├── init_qdrant.py                     # DB initialization
├── main.py                            # Entry script
├── requirements.txt
├── pyproject.toml
├── .env / .env.example
├── CHANGELOG.md
├── LICENSE
└── README.md
```

### 🧠 Design Philosophy

* **app/** → Core AI system (pipeline + memory + reasoning)
* **configs/** → Environment & configuration management
* **data/** → Raw + processed multimodal data
* **models/** → Local GGUF models
* **offload/** → CPU optimization for large models
* **scripts/** → Utility and automation scripts
* **tests/** → Comprehensive module-level testing

This structure ensures:

* Modular design
* Scalability
* Clear separation of concerns
* Production-grade maintainability


---

## 🔄 RAG Pipeline

Query → Embed → Retrieve → Context → Memory → Reasoning → LLM → Answer

---

## 🎯 Multimodal Pipeline

Input → Text Conversion → Chunk → Embed → Store → Retrieve

---

## 📊 Evaluation & Monitoring (Planned)

* RAGAS
* DeepEval
* Langfuse
* Prometheus + Grafana

---

## 🚀 Deployment (Planned)

* Docker + AWS EC2
* S3 storage
* CI/CD (GitHub Actions)

---

## 📈 Future Improvements

* Reranker (cross-encoder)
* Smart memory prioritization
* Session-based optimization
* Observability dashboards
* Cloud deployment

---

## 👨‍💻 Author

**VK**
Aspiring GenAI Engineer

---

## 🎯 What This Project Demonstrates

* End-to-end multimodal RAG system
* Memory-integrated AI architecture
* Reasoning-enhanced LLM pipeline
* FastAPI + Gradio full-stack system
* Production-grade GenAI design


