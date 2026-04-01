from app.vectorstore.qdrant_store import QdrantVectorStore
from app.embeddings.text_embedder import TextEmbedder


class Retriever:

    def __init__(self):

        self.vector_store =  QdrantVectorStore()
        self.embedder = TextEmbedder()

    # -----------------------
    # NEW: QUERY REWRITING
    # -----------------------
    def _rewrite_query(self, query: str) -> str:
        q = query.lower()

        if "video about" in q or "What is the video about" in q:
            return "motivation speech encouragement message meaning topic"
        
        if "what is happening" in q:
            return "scene action activity people doing"
        
        if "who" in q:
            return "person people man woman speaker"
        
        if "describe" in q:
            return "description scene objects people environment"
        
        return query
    
    # ---------------------------
    # MAIN RETRIEVAL 
    # ---------------------------
    def retrieval(self, query: str, top_k: int = 5, source: str = None):

        # Step 1: Rewrite query 
        rewritten_query = self._rewrite_query(query)
        print("DEBUG QUERY:", rewritten_query)

        query_vector = self.embedder.embed_query(rewritten_query)


        # Step 2: Retrieve more results
        results = self.vector_store.search_text(
            query_vector,
            limit=top_k * 2, 
            source_filter=source
        )

        # Step 3: Separate modalities
        audio_docs = []
        frame_docs = []
        text_docs = []

        for r in results:
            modality = r["metadata"].get("modality", "text")

            if modality in ["audio", "video_audio"]:
                audio_docs.append(r)
            
            elif modality == "video_frame":
                frame_docs.append(r)
            
            else:
                text_docs.append(r)

     
        # Step 4: Balanced Selection
        if audio_docs:
            final_results = (
                audio_docs[:3] +
                frame_docs[:2] +
                text_docs[:1]
            )
        elif frame_docs:
            final_results = frame_docs[:top_k]

        else:
            final_results = text_docs[:top_k]


        # Step 5: Debug 
        print("\n=== FINAL RETRIEVAL ===")
        for r in final_results:
            print(r["metadata"].get("modality"), "|", r["text"])
        print("========================\n")

        return final_results