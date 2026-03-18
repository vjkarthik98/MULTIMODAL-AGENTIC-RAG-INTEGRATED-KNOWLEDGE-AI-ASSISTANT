from app.retrieval.retriever import Retriever

class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()

    def run(self, query: str):
        
        documents = self.retriever.retrieval(query)

        return documents