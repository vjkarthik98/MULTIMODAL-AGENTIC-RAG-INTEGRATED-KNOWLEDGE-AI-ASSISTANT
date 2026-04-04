"""
reasoning_engine.py

Handles multimodal reasoning before LLM generation.
Combines:
- user query
- retrieved context
- memory context

This is the intelligence layer of the system.
"""

from typing import List, Dict


class ReasoningEngine:

    def __init__(self, llm):
        self.llm = llm

    def build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = ""
    ) -> str:
        """
        Construct a structured reasoning prompt.
        """

        # Step 1: Prepare knowledge context
        knowledge_context = "\n".join(
            [doc.get("text", "") for doc in retrieved_docs]
        )

        # Step 2: Build structured prompt
        prompt = f"""
You are an advanced multimodal AI assistant.

Your task is to:
- Understand the user's intent
- Use past conversation (if relevant)
- Use retrieved knowledge
- Reason step-by-step before answering

Conversation Context:
{memory_context}

Knowledge Context:
{knowledge_context}

User Query:
{query}

Instructions:
1. Analyze the query carefully
2. Use relevant memory if needed
3. Use retrieved knowledge
4. Think step-by-step (internal reasoning)
5. Provide a clear, final answer

Answer:
"""

        return prompt.strip()

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = ""
    ) -> str:
        print("REASONING ENGINE CALLED")
        """
        Generate final answer using reasoning prompt.
        """

        prompt = self.build_prompt(
            query,
            retrieved_docs,
            memory_context
        )

        answer = self.llm.generate(prompt)

        return answer.strip()
    