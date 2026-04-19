class PromptBuilder:
    def build_prompt(self, query: str, context: str) -> str:
        return f"""
<s>[INST]

You are a highly reliable AI assistant.

====================
CONTEXT
====================
{context}

====================
USER QUERY
====================
{query}

===================
INSTRUCTIONS
===================

1. Answer ONLY using the provided context
2. Do NOT hallucinate or assume missing facts
3. If the answer is not in context -> say "I don't know"
4. Prefer factual, concise, and structured responses
5. If multiple pieces of information exist -> combine them logically
6. For multimodal content:
    - Audio -> focus on spoken information
    - Visual -> describe scene or objects
    - Text -> prioritize factual content

===================
OUTPUT FORMAT
===================

Answer:
<final answer>

Confidence:
<number between 0 and 1>

===================

[/INST]
""".strip()
    
