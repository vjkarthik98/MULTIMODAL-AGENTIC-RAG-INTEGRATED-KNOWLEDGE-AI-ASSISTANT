class PromptBuilder:

    def build_prompt(self, query, context):
        # Final prompt
        prompt = f"""<s>[INST]
Context:
{context}

Q: {query}
A:
[/INST]"""
        
        return prompt