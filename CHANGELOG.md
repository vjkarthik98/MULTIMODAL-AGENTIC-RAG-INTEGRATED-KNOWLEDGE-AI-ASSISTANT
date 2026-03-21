# Changelog

All notable changes to this project will be documented on this file.
The format follows Keep a Changelog and Semantic Versioning.


---

# [0.1.0] - Initial Setup

### Added

- Github repository with development workflow
- Protected main branch
- Project folder architecture
- Python package structure for rag_system
- Environment configuration template (.env.example)
- Dependency management (requirements.txt)
- Python packaging (pyproject.toml)
- VERSION file for semantic versioning

# [0.2.0] - FastAPI Integration

### Added 

- FastAPI backend for RAG system
- API endpoints for query handling
- Integration of RAG pipeline with API
- End-to-End system with Qdrant + Ollama

# [0.3.0] - Multimodal Ingestion
### Added
- Multimodal ingestion pipeline (text, image, audio, video)
- Faster-Whisper integration for audio transcription
- OCR-based image text extraction
- Video-to-audio transcription pipeline
- Unified ingestion router

### Improved
- Metadata-aware ingestion schema
- Standardized document structure (text + metadata)
- Modula ingestion architecture

# [0.4.0] - Chunking Integration
### Added 
- Recursive chunking with overlap
- Batch embedding pipeline
- Qdrant vector database integration
- Retrieval system with semantic search
- End-to-end ingestion pipeline (chunk -> embed -> store)
- FastAPI upload endpoint for document ingestion
- Full RAG pipeline (retrieve + generate)

### Improved
- Embedding performance using batching
- Modular architecture (utils, ingestion, embeddings, vectorstore)
- Automatic Qdrant collection creation
- Context formatting for better LLM responses

### Bug Fixed
- Qdrant API compatibility issues ('query_points')
- Retrieval method mismatches
- Vector Store integration bugs