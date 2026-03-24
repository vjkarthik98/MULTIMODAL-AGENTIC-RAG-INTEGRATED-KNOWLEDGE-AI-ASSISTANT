# 🧠 Multimodal RAG Knowledge Assistant

## 🚀 Overview

A production-grade **Multimodal Retrieval-Augmented Generation (RAG) system** built with FastAPI, Qdrant, and open-source LLMs.

This system enables intelligent querying over **text, PDFs, images, audio, and video**, transforming unstructured data into a searchable knowledge base.

### 🔥 Key Highlights

* 🧠 Multimodal ingestion (OCR + Speech-to-Text + Video processing)
* ⚡ Optimized RAG pipeline (chunking + embedding + retrieval)
* 🗄️ Qdrant vector database for semantic search
* 🚀 FastAPI backend with modular architecture
* 🧩 Extensible design for LLMOps, monitoring, and deployment

> Designed using production-grade AI system architecture patterns used in modern RAG startups.
## ❗ Problem Statement

Modern organizations deal with large volumes of unstructured data such as documents, images, audio, and videos.

Traditional search systems fail to understand semantic meaning, and most RAG implementations are limited to text-only pipelines with poor chunking strategies and weak retrieval accuracy.

As a result:

* Relevant information is missed during retrieval
* Context is lost due to improper chunking
* Multimodal data remains unused
* Systems are not scalable or production-ready


## 💡 Solution

This project implements a **production-grade Multimodal RAG system** that transforms unstructured data into a structured, searchable knowledge base.

Key solutions:

* Converts all modalities (text, PDF, image, audio, video) into unified text representation
* * Applies **recursive chunking with overlap** to preserve semantic context
* Uses **optimized embeddings** for accurate semantic retrieval
* Stores chunk-level vectors in Qdrant for fine-grained search
* Exposes the system via FastAPI for real-world usage

This design ensures high retrieval accuracy, scalability, and extensibility for real-world AI applications.


## 🏗️ System Architecture

The system is designed using a layered architecture similar to production AI systems.

### 🔷 High-Level Flow

User → API Layer → RAG Pipeline → Vector DB → LLM → Response

### 🧱 Architecture Layers

1. **Frontend Layer**

   * User interaction (UI / API clients)

2. **API Layer (FastAPI)**

   * Handles requests and routing
   * Connects user queries to RAG pipeline

3. **AI Pipeline**

   * Query embedding
   * Vector search (Qdrant)
   * Context retrieval (top-k chunks)
   * Prompt construction
   * LLM response generation

4. **Data Layer**

   * Qdrant → vector storage
   * Stores chunk-level embeddings with metadata

5. **Infrastructure Layer (Planned)**

   * Redis → caching
   * MongoDB → conversation memory
   * Monitoring tools → observability

This modular design ensures scalability, maintainability, and production readiness.


## ⚙️ Tech Stack

The project is built using a modern AI system stack inspired by production RAG architectures.

### 🧠 AI / ML

* Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
* LLM: Mistral (via Ollama)
* RAG Pipeline: Custom implementation

### ⚡ Backend

* FastAPI (API layer)
* Python (core logic)

### 🗄️ Data & Storage

* Qdrant (vector database)
* Chunk-level embedding storage with metadata

### 🔄 Processing

* Text processing & chunking (recursive splitting)
* Multimodal preprocessing:

  * PDF → text extraction
  * Image → captioning
  * Audio → speech-to-text
  * Video → frame extraction + captioning

### 🧱 Architecture & Engineering

* Modular backend design (routes → services → core)
* Batch embedding for performance optimization
* Metadata-aware retrieval

### 🚀 Future Stack (Planned)

* Redis (caching)
* MongoDB (conversation memory)
* AWS S3 (document storage)
* Prometheus + Grafana (monitoring)
* MLflow (experiment tracking)


## ✨ Features

### 🔍 Core RAG Capabilities

* Semantic search using vector embeddings
* Context-aware answer generation using LLM
* Top-k retrieval for relevant context selection

### 🧠 Advanced Retrieval Optimization

* Recursive chunking with overlap for context preservation
* Chunk-level embedding storage (not document-level)
* Batch embedding for improved performance

### 🎯 Multimodal Intelligence

* Text and PDF ingestion
* Image understanding via captioning
* Audio processing via speech-to-text
* Video understanding via frame extraction and captioning

### 🌐 API-Based System

* FastAPI backend for real-time querying
* Modular route-based architecture
* Ready for frontend or external integration

### 🧱 Scalable Architecture

* Modular design (ingestion, embeddings, retrieval, API)
* Metadata-aware storage for filtering and extensibility
* Designed for multi-user and multi-document support

### ⚙️ Engineering Best Practices

* Clean separation of concerns
* Batch processing for efficiency
* Extensible pipeline design


## 📂 Project Structure

## 📂 Project Structure

```bash
project_root/
│
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── rag_routes.py      # FastAPI routes
│   │
│   ├── ingestion/                # Data ingestion (text, PDF, etc.)
│   ├── embeddings/               # Embedding logic
│   ├── vectorstore/              # Qdrant integration
│   ├── utils/                    # Chunking, preprocessing utilities
│   ├── core/                     # Config, settings
│
├── src/
│   └── rag_system/
│       ├── pipeline/             # RAG pipeline (retrieve → generate)
│       ├── prompt/               # Prompt engineering
│       ├── generation/           # LLM interaction
│
├── tests/                        # Test cases
├── main.py                       # FastAPI entry point
├── requirements.txt
└── README.md
```

### 🧠 Design Philosophy

* **app/** → Handles infrastructure & APIs
* **src/** → Contains core AI logic (RAG pipeline)




This separation ensures:

* Clean architecture
* Scalability
* Easy maintainability


## 🔄 RAG Pipeline

The system follows a Retrieval-Augmented Generation (RAG) pipeline for answering user queries.

### 🔁 Pipeline Flow

1. **User Query**

   * Input question from user

2. **Query Embedding**

   * Convert query into vector representation

3. **Vector Search (Qdrant)**

   * Perform similarity search over stored embeddings

4. **Top-K Retrieval**

   * Retrieve most relevant chunks (not full documents)

5. **Context Construction**

   * Combine retrieved chunks into a structured prompt

6. **LLM Generation**

   * Generate answer using Mistral (via Ollama)

7. **Response Output**

   * Return answer along with relevant sources

---

### ⚡ Key Optimizations

* Chunk-level retrieval instead of document-level
* Recursive chunking with overlap
* Batch embedding for performance
* Metadata-aware filtering (extensible)

---

### 🎯 Why This Matters

This approach improves:

* Retrieval accuracy
* Context relevance
* Answer quality


## 🎯 Multimodal Capabilities

The system supports ingestion and understanding of multiple data modalities by converting them into a unified text representation for RAG processing.

### 🧾 Supported Modalities

#### 📄 Text & PDF

* Direct text ingestion
* PDF parsing and text extraction

#### 🖼️ Image

* Optical Character Recognition (OCR) using Tesseract
* Extracted text used for embedding and retrieval

#### 🎤 Audio

* Speech-to-text conversion using Whisper
* Transcribed text integrated into RAG pipeline

#### 🎥 Video

* Audio extraction from video
* Speech-to-text processing
* Converted into searchable text format

---

### 🔄 Unified Processing Flow

All modalities follow a common pipeline:

Input (any modality) → Text Conversion → Chunking → Embedding → Vector DB → Retrieval

---

### ⚡ Design Advantage

* Single unified pipeline for all data types
* Scalable to new modalities (e.g., vision LLMs)
* Consistent retrieval across heterogeneous data sources


## 🌐 API Layer (FastAPI)
The system exposes its functionality through a FastAPI backend, enabling real-time interaction with the RAG pipeline.

### 🔌 Key Endpoints

#### 🔍 Query Endpoint

```http id="qv4wsp"
POST /rag/query
```

**Request:**

```json id="v6qozx"
{
  "query": "What is artificial intelligence?"
}
```

**Response:**

```json id="qv8k91"
{
  "answer": "Artificial intelligence is ...",
  "sources": ["document1", "document2"]
}
```

---

#### 📁 File Upload Endpoint

```http id="6l9vav"
POST /upload/file
```

* Supports multimodal file ingestion
* Automatically routes based on file type
* Integrates into RAG pipeline

---

### 🧠 Backend Design

* Modular route-based architecture
* Separation of concerns:

  * Routes → API layer
  * Services → RAG logic
  * Core → configuration
* Easy integration with frontend or external systems

---

### ⚡ Why This Matters

* Converts RAG into a real backend system
* Enables deployment on cloud platforms
* Supports scalable AI applications

## 🧪 Installation & Setup
### 🔧 Prerequisites

* Python 3.10+
* Docker (for Qdrant)
* FFmpeg (for audio/video processing)
* Tesseract OCR (for image processing)
* Ollama (for running LLM locally)

---

### 📦 Clone Repository

```bash id="f3n2lm"
git clone https://github.com/vjkarthik98/multimodal-rag-assistant.git
cd multimodal-rag-assistant
```

---

### 🧱 Create Virtual Environment

```bash id="0d2kqs"
python -m venv rag_env
source rag_env/bin/activate      # Linux / Mac
rag_env\Scripts\activate         # Windows
```

---

### 📥 Install Dependencies

```bash id="8k3lmn"
pip install -r requirements.txt
```

---

### 🗄️ Run Qdrant (Docker)

```bash id="7lmv9a"
docker run -p 6333:6333 qdrant/qdrant
```

---

### 🧠 Run Ollama (LLM)

```bash id="l9a2ks"
ollama run mistral
```

---

### 🚀 Start FastAPI Server

```bash id="n2ksla"
uvicorn main:app --reload
```

---

### 🌐 Access API

* Swagger UI: http://127.0.0.1:8000/docs
* Test endpoint: http://127.0.0.1:8000/rag/test

---

### 🧪 Example Query

```json id="9sk2la"
{
  "query": "Explain artificial intelligence"
}
```




## 📊 Evaluation & Monitoring

The system is designed with evaluation and observability in mind to ensure reliability and performance.

### 📏 Evaluation Metrics (Planned)

* **Answer Relevance** — measures how well the response matches the query
* **Context Recall** — evaluates whether relevant chunks are retrieved
* **Faithfulness** — checks if answers are grounded in retrieved context

### 🧪 Evaluation Frameworks

* RAGAS — for automated RAG evaluation
* DeepEval — for LLM response quality assessment

---

### 📡 Monitoring & Observability (Planned)

* **Langfuse** — tracing LLM calls and prompt performance
* **Prometheus** — system-level metrics (latency, throughput)
* **Grafana** — visualization dashboards

---

### ⚡ Why This Matters

* Detects hallucinations and retrieval failures
* Improves system reliability over time
* Enables production-grade AI monitoring



## 🚀 Deployment (Planned)

The system is designed to be deployed using a scalable cloud architecture.

### ☁️ Deployment Strategy

* **Compute**: AWS EC2 (Dockerized services)
* **Storage**: AWS S3 for document storage
* **Containers**:

  * FastAPI backend
  * Qdrant vector database
  * Redis (caching layer)

---

### 🔄 CI/CD Pipeline (Planned)

* GitHub Actions for automated workflows
* Steps:

  * Run tests
  * Run evaluation pipeline
  * Build Docker images
  * Deploy to cloud

---

### 🧱 Deployment Architecture

User → FastAPI → RAG Pipeline → Qdrant → LLM
↓
Monitoring Stack

---

### ⚡ Goals

* Low-cost deployment
* Scalable architecture
* Production-ready system design


## 📈 Future Improvements
The system is designed to be extensible and can be enhanced with additional capabilities:

* 🔁 Reranking models (e.g., BGE reranker) for improved retrieval accuracy
* 🧠 Conversation memory (short-term + long-term)
* 🧑‍🤝‍🧑 Multi-tenant support with user-based data isolation
* ⚡ Caching layer using Redis for faster responses
* 📊 Monitoring and observability (Prometheus, Grafana, Langfuse)
* 🧪 Evaluation frameworks (RAGAS, DeepEval)
* ☁️ Cloud deployment on AWS (EC2 + S3 + Docker)
* 🔐 Authentication and authorization (JWT-based access)

These improvements align the system with production-grade AI platforms used in industry.


## 👨‍💻 Author
**VK**
Aspiring GenAI Engineer | Building production-ready AI systems

---

### 🎯 What This Project Demonstrates

* End-to-end RAG system design (retrieval → generation → API)
* Multimodal AI pipeline (text, image, audio, video)
* Backend engineering with FastAPI
* Vector database integration (Qdrant)
* Retrieval optimization (chunking + embeddings)
* System design aligned with real-world AI architectures

---

### 📬 Contact

* LinkedIn: *P*
* GitHub: *www.github.com/vjkarthik98*





