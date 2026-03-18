class PromptBuilder:

    def build_prompt(self, query, contexts):

        formatted_context = ""

        for i, context in enumerate(contexts, start=1):
            formatted_context += f"Context {i}:\n{context}\n\n"

        
        prompt = f"""
You are an AI assistant answering questions using provided context.

Only use the context below to answer the question.
If the answer is not in the context, say "I don't know".

{formatted_context}

Question:
{query}

Answer:
"""
        
        return prompt