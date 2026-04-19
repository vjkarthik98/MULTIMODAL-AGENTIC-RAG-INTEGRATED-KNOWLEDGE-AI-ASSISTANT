from typing import List, Dict, Any
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class ReasoningEngine:

    def __init__(self, llm):
        self.llm = llm

    # Main API
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        memory_context: str = ""
    ) -> Dict[str, Any]:

        try:
            logger.info("[ReasoningEngine] Start reasoning")

            # STEP 1: Prepare Context
            knowledge = self._prepare_knowledge(retrieved_docs)
            memory = self._prepare_memory(memory_context)

            # STEP 2: Build Prompt
            prompt = self._build_prompt(query, knowledge, memory)


            # STEP 3: Generate
            response = self.llm.generate(prompt)

            # STEP 4: Parse
            parsed = self._parse_response(response)

            logger.info("[ReasoningEngine] Completed")

            return parsed
        
        except Exception as e:
            logger.error(f"[ReasoningEngine] Failed | {str(e)}")

            return {
                "answer": "I couldn't generate a reliable answer.",
                "confidence": 0.2,
                "sources_used": 0
            }
        
    # Context Preparation
    def _prepare_knowledge(self, docs: List[Dict]) -> str:

        if not docs:
            return ""
        
        selected = []

        for d in docs[:5]:
            text = d.get("text", "").strip()

            if not text:
                continue

            source = d.get("source", "unknown")

            selected.append(f"[Source: {source}] {text[:300]}")

        return "\n\n".join(selected)
    
    def _prepare_memory(self, memory: str) -> str:

        if not memory:
            return ""
        
        return memory.strip()[:1000]

    # Prompt
    def _build_prompt(self, query: str, knowledge: str, memory: str) -> str:

        return f"""
You are a highly reliable AI assistant. 

Your goal:
- Answer ONLY using provided context
- Do NOT hallucinate
- If information is missing, say "I don't know"

-------------------------------
MEMORY:
{memory}

------------------------------
KNOWLEDGE:
{knowledge}

-----------------------------
USER QUERY:
{query}

----------------------------
INSTRUCTIONS:

1. Use MEMORY if relevant
2. Use KNOWLEDGE as primary source
3.Do NOT invent facts
4. If unsure -> say "I don't know"
5. Keep answer concise and factual

------------------------------------
OUTPUT FORMAT (STRICT):

Answer:
<final answer>

Confidence:
<number between 0 and 1>

Source Used:
<number of sources used>
""" 
    # Response Parser
    def _parse_response(self, text: str) -> Dict[str, Any]:

        try:
            answer = ""
            confidence = 0.7
            sources = 0

            lines = text.split("\n")

            for line in lines:
                line = line.strip()

                if line.lower().startswith("answer"):
                    answer = line.replace("Answer:", "").strip()

                elif line.lower().startswith("confidence:"):
                    try:
                        confidence = float(line.split(":")[1].strip())

                    except:
                        confidence = 0.5
                elif line.lower().startswith("sources used:"):
                    try:
                        sources = int(line.split(":")[1].strip())
                    except:
                        sources = 0
            if not answer:
                answer = text.strip()

            return {
                "answer": answer,
                "confidence": confidence,
                "sources_used": sources
            }
        
        except Exception as e:
            logger.error(f"[ReasoningEngine] Parse failed | {str(e)}")

            return {
                "answer": text.strip(),
                "confidence": 0.5,
                "sources_used": 0
            }