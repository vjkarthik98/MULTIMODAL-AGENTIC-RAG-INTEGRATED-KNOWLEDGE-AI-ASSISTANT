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

# [0.5.0] - GGUF Model Integration 
### Added
- GGUF model integration using llama.cpp (CPU-based inference)
- Streaming response support using FastAPI StreaminResponse
- New `/rag/query/stream` endpoint for real-time token output
- Modular LLM wrapper (`gguf_model.py`)

### Changed
- Replaced HuggingFace/Ollama with GGUF-based inference
- Updated RAG pipeline to use local quantized models
- Improved prompt formatting for Mistral Instruct

### Fixed
- Upload endpoint routing issue
- Multipart file upload handling
- Qdrant collection initialization flow
- UTF-8 decoding issue for non-text files (handled via testing approach)

### Notes
- Optimized for CPU environments (no GPU dependency)
- Designed for lightweight deployment and scalability

# [0.6.0] - Improved RAG Pipeline

### Added
- Chunk-based ingestion pipeline
- Batch embedding for documents
- Unique UUID-based vector storage in Qdrant
- Retriever with configurable top-k
- Structured context formatting for LLM

### Improved
- Token usage optimization for GGUF models
- Prompt design for better grounding
- Duplicate document removal in retrieval

### Fixed
- Context window overflow errors
- UUID generation bug in Qdrant storage
- Prompt NoneType crash issue

### Notes
- System now supports production-level RAG pipeline
- Fully optimized for CPU-based local inference

# [0.7.0] - Production-grade ingestion pipeline

### Added 
- End-to-end ingestion pipeline orchestration (process_file)
- Structured API response (status, details, chunks)
- Logging system for pipeline observability

### Improved
- Chunk -> Embed -> Stored pipeline consistency
- Qdrant storage with document_id grouping
- Metadata strucutre for better retrieval

### Fixed 
- Ingestion returning incorrect format
- Embedding validation issues
- Duplicate / untraceable vector storage

### Validation
- Added checks for:
    - Empty documents
    - Embedding failure
    - Invalid vector formats

# [0.8.0] - Text and Image (Ingestion + Query)
### Added
- Image ingestion pipeline using CLIPVisionMOdelWithProjection
- Image query pipeline using CLIPTextModelWithProjection
- Consistent dimensional embedding across ingestion and query

### Fixed
- Dimension mismatch issue (512 vs 768) during image query

### Notes
- Qdrant collection standardized to 384(Text) and 768(Image) vector size
- Foundation ready for multimodal expansion (audio/video)

# [0.9.0] - Audio Intelligence Upgrade
### Features
- Added audio ingestion using faster-whisper
- Converted audio to text segments for embedding
- Implemented audio query endpoint (/rag/query/audio)
- Enabled full RAG pipeline for audio queries

### Improvements
- Segment-level audio chunking
- Added timestamps (start_time, end_time) in metadata
- Context enriched with audio time references

### Fixes
- Fixed retriever (removed invalid search() call)
- Ensured pipeline consistency for all modalities

### Result
- Full multimodal RAG (text, image, audio)
- Production-grade ingestion + query pipeline

# [0.10.0] - Multimodal Video Rag
### Added 
- Video ingestion pipeline
- Frame extraction + BLIP captioning
- Audio extraction + transcription (faster-whisper)
- Multimodal embeddings (audio + frame)
- Unified storage in Qdrant
- Video query endpoint
- Retrieval balancing (audio + frame)

### Improved
- Query rewriting for better semantic retrieval
- Context aggregation for better LLM understanding
- Prompt optimization for meaning-based answers

### Fixed
- Audio chunking issue (single segment bug)
- FFmpeg path resolution
- Tuple vs dict retrieval bug
- Metadata loss in Qdrant payload

### Known Issues
- LLM sometimes prefers visual description over audio meaning
- Needs reranking / weighting improvement

# [0.11.0] - UI
### Features
- UI with Multi-chat sessions
- Streaming responses (real-time token generation)
- Multimodal upload support (PDF, Image, Audio, Video)
- File ingestion pipeline integrated with UI
- Session-based chat switching

### RAG Improvements
- Stable retrieval pipeline (fixed modality handling)
- Context construction improvements
- Source attribution in responses

### Bug Fixes
- Fixed Qdrant collection mismatch issues
- Fixed embedding consistency (query vs document)
- Fixed streaming pipeline (backend + frontend sync)
- Fixed Gradio message format issues
- Fixed chat state synchronization bug

### Internal
- Refactored retriever logic
- Improved vector store insertion reliability
- Cleaned pipeline structure

# [0.12.0] - Memory System Integration
### Features
- Added Redis-based short-term conversational memory
- Added MongoDB-based persistent memory storage
- Implemented session-based memory handling (multi-user support)
- Integrated memory into RAG pipeline
- Enabled memory support in Gradio UI (Non-Stream mode)

### Improvements
- Clean Separation of memory layers:
    - Redis -> short-term context
    - MongoDB -> long-term storage
    - Qdrant -> knowledge retrieval
- Improved pipeline structure for memory injection
- Fixed session handling across API and UI

### Fixes
- Fixed Redis key formatting issue (Whitespace bug)
- Fixed Gradio streaming vs non-stream mismatch
- Fixed memory not storing from UI requests
- Corrected pipeline memory storage order (post-generation)

### Infr
- Dockerized Redis and MongoDB
- Added persistent container strategy (restart policies)

# [0.13.0] - Summarization of Redis Memory
### Added 
- MemoryManager for automatic summarization
- LLM-based conversation summarization
- Conversation history injection into RAG pipeline

### Improved
- Prompt now includes conversation history
- Gradio UI supports conversational flow

### Fixed
- Redis key mismatch bug
- Memory overwrite issue
- Gradio message format error
- LLM response not returned properly

# [0.14.0] - Smart Memory Optimization
### Added
- Memory formatter
- Semantic memory filtering
- Memory summarization (LLM-based)
- Memory fusion Layer

### Improvement
- Designed token-efficient, context-aware memory system

# [v0.15.0] - Systerm Integration & Stabilization
### Features
- Image Captioning using BLIP
- Audio Transcription pipeline
- Video frame + audio processing
- Unified embedding pipeline (SentenceTransformer - 384 dim)
- Qdrant Vector database integration
- Redis (short-term) + MongoDB(long-term) memory
- Dynamic, modality-aware prompt system

### Improved
- Retrieval accuracy using reranker
- Reduced hallucinations with retrieval guard
- Unified embedding space across modalities
- Better query relevance and context handling

### Fixed
- BLIP captioning failures(generate()bug)
- Image ingestion issues (PIL, RGB, EXIF)
- Vector dimension mismatch (768 vs 384)
- Retrieval and reranker execution issues

# [v0.16.0] - Multimodal Intelligence & Reasoning
### Features
- Reasoning Engine
- Query Decomposition
- Mutli-Query Retrieval
- Result Fusion & Ranking Layer
- Diversity Filtering for context optimization

### Improved
- Retrieval quality via multi-hop search
- Answer quality with structured reasoning
- Context relevance with fusion + filtering
- Memory + reasoning integration


# [v0.17.0] - Agentic Pipeline + Model Loader + Full Observability
### Features
- Introduced AgentController for intelligent query routing
- Added decision-based execution (Multimodal vs standard pipeline)
- Added websearch tool for enhanced latest output
- Enabled dynamic handling of image, audio, video and text queries

### Improved
- Centralized model management via ModelLoader
- Eliminated scattered model initialization across modules
- Centralised logging system via get_logger
- Replaced all print statemens with structured logging

# [v0.18.0] - Hybrid Retrieval + Reranking Stabilization
### Features
- Implemented BM25-based keyword retrieval using rank-bm25
- Integrated semantic vector search using existing embedding pipeline
- Built HybridRetriever to combine keyword + semantic results
- Added CrossEncoder-based reranker for result refinement
- Improved retrieval precision and reduced irrelevant chunks
- Multimodal Ingestion Improvements
    -> PDF: Text Extraction + Image Extraction + Table Extraction
    -> Word: Text Extraction + Image Extraction + Table Extraction
    -> Excel: Table Extraction with structured conversion
- Metadata Standardization


# [v0.19.0] - Multimodal system Refactor & Architecture Strengthening
### Features
- Standardized multimodal ingestion across text, document, image, audio, video
- Introduced structured document schema with doc_id, session_id, file_hash
- Enabled modality-aware outputs (caption, OCR, speech, frame, tables)
- Added enriched metadata(timestamps, pages, segments, source tracking)
- Improved audio(Whisper) and video(audio + frame fusion) pipelines
- Enhanced image pipeline with capiton + OCR dual representation
- Introduced ModelLoader with lazy loading to initialize models only on first use
- Added centralized caching to prevent redundant model loads across modules
- Introduced Config for centralized Details of Important details

### Improved
- Refactored ingestion to produce structured blocks instead of flat text
- Strengthened document parsing (PDF, WORD, Excel) with multimodal extraction
- Aligned all modalities for consistent embedding + retrieval readiness
- Upgraded memory filtering with embedding reuse + recency/role scoring
- Improved pipeline modularity (ingestion vs query separation)
- Standardized logging, error handling and session tracking across modules
- Added validation and fallback handling in ingestion workdflows
- Reduced startup latency by deffering heavy model initialization
- Improved memory efficiency with on-demand model lifecycle management
- Standardized model access across pipeline via unified loader interface


# [v0.20.0]- Deterministic Multimodal RAG Stabilization & Agent Control Hardening
### Features
- Strict grounding enforced (LLM answers only from retrieved context)
- Multi-user session isolation across retrieval, memory, and vector store
- Hybrid execution support (RAG + Search)
- Intent-aware agent routing with multi-query handling
- Parallel sub-query execution for complex queries
- Structured API response (answer, sources, confidence, trace)
- Modality-aware retrieval (text, image, audio, video)
- Video frame-level retrieval integration
- Fail-safe agent fallback hierarchy

### Improved
- Retrieval: Fixed BM25 flow, improved hybrid ranking, added score filtering
- Agent: Reduced randomness, improved routing stability, added trace logging
- RAG Pipeline: Better context filtering, reduced hallucination, optimized token usage
- Ingestion: Improved chunking, standardized metadata, stronger validation
- Multimodal: Enhanced captioning, transcription, and modality alignment
- Memory: Strict session isolation, improved conversation handling
- Prompting: Enforced context-only answers, improved structure
- Model Loading: Centralized loading with retry and timeout handling
- API & Logging: Structured responses, better error handling, improved observability

### Fixed 
- BM25 indexing and retrieval issues
- Duplicate and low-quality chunk retrieval
- Incorrect agent routing and fallback behavior
- Metadata loss in multimodal ingestion
- Context overflow and embedding inconsistencies
- Session leakage across memory and retrieval
- Hybrid retriever score normalization bug
- Edge-case ingestion and pipeline failures

# [v0.21.0] - Production Hardening, Multimodal Edge-Case Robustness & Test Foundation

### Features
- Bounded agent execution (max-steps + wall-clock timeout + token budget)
- Tenant isolation hook via Qdrant `user_id` payload filter (typed, not string-built)
- Circuit breaker (`_CircuitBreaker`) on Qdrant calls with half-open probe
- GDPR purge path across Qdrant + Redis + Mongo
- Hallucination guard + numeric-faithfulness check in reasoning engine
- Temporal-anchor boost for time-sensitive queries (FY/quarter/prior-year)
- MMR diversity in retriever, modality-aware reranking
- Standardised query response schema (answer, sources, confidence, trace)

### Improved
- Multilingual support removed (deferred); pipeline simplified to English path
- Full `.env` ↔ `config.py` alignment; zero hardcoded literals in pipeline
- Text/PDF/Word/Excel ingestion hardened against broken + edge-case files
- Image/Audio/Video ingestion hardened (solid-color detect, OCR repair, frame dedup via pHash)
- Retrieval + RAG pipeline end-to-end repair (Phase 24.7)
- Startup latency cut (~25s → ~7s) via lazy device manager + deferred model loads
- Per-model device routing (CPU/CUDA/MPS) with profile auto-resolve
- Lifecycle management for models, infra and pipelines (lazy singletons + health)
- Memory layer: sliding window + dedup + role/recency-weighted fusion
- Caption sanitizer + prompt-injection scrub on BLIP/CLIP/web-search inputs

### Fixed
- Qdrant payload index on `user_id` for multi-user filtering
- Section-aware chunking (preserves structure for temporal queries)
- Warmup race + startup crashes on missing optional deps
- Embedding cache key drift across modalities
- Empty-context path now raises `EMPTY_CONTEXT_NO_DOCUMENTS_RETRIEVED` instead of silent stub

