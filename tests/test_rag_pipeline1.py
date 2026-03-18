from src.rag_system.pipeline.rag_pipeline import RAGPipeline 

pipeline = RAGPipeline()

query = "What is artificial Intelligence?"

result = pipeline.run(query)

print("\nAnswer: \n")
print(result["answer"])

print("\nSources:\n")
for s in result["sources"]:
    print("-", s)
