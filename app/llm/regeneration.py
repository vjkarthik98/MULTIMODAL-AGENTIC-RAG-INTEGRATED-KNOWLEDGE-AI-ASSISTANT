"""What makes a user-requested regeneration actually different.

The regenerate button re-runs the whole pipeline — retrieval, rerank, fusion,
prompt build, generation — and no cache is read (`no_cache=True`). That was
never the problem. The problem is that every stage of that pipeline is
deterministic: the same query retrieves the same chunks in the same order,
builds the same prompt, and decodes it at temperature 0.0-0.1, which is
argmax at every step. Re-running a deterministic function returns the same
value, so the user pressed "regenerate" and got the previous answer back,
character for character.

Determinism is the right default for a finance RAG — it's what makes an
answer reproducible and an eval baseline meaningful — so this does not
loosen it globally. It applies, only when the user explicitly asks for a
different answer:

  1. a temperature floor (settings.LLM_TEMPERATURE_REGENERATE), so the
     sampler can actually leave the greedy path, and
  2. a fresh random seed, so two regenerations of the same prompt at the
     same temperature don't collapse back onto the same trajectory, and
  3. a prompt directive telling the model its previous answer was rejected
     and to answer the same evidence differently and more completely.

(3) matters as much as (1) and (2): raising temperature alone yields a
reworded version of the same answer, which is not what someone pressing
regenerate wants — they want the model to try harder. It is also written to
push toward *more* grounding, not less, because the one thing a regenerate
must never do is invent a figure to look different from the last attempt.
"""

from __future__ import annotations

import secrets

from app.core.config import settings

# 31-bit: fits every downstream seed field (llama.cpp's uint32, the
# OpenAI-compatible server's int) without risking a negative or overflow.
_MAX_SEED = 2**31 - 1


def regeneration_sampling(base_temperature: float) -> tuple[float, int]:
    """Return (temperature, seed) to use for a regeneration.

    The temperature is a floor, not an override: a query type that already
    samples hotter than the floor (generative/comparative) keeps its own
    value rather than being cooled down to it.
    """
    temperature = max(float(base_temperature), float(settings.LLM_TEMPERATURE_REGENERATE))
    return temperature, secrets.randbelow(_MAX_SEED)


# Placed before the QUERY block by prompt_builder.build_prompt(regenerate=True)
# — never after the answer cue, where the model would continue this text
# instead of obeying it.
REGENERATE_DIRECTIVE = (
    "REGENERATE:\n"
    "The previous answer to this question was rejected by the user. Answer "
    "again from the SAME evidence above — do not reuse the previous wording. "
    "Restructure the answer, state the specific figures, dates, periods and "
    "entity names the evidence supports, and cover any part of the question "
    "the previous attempt left out. Do NOT add any fact that is not in the "
    "evidence above; if the evidence does not answer part of the question, "
    "say so plainly instead of filling the gap.\n\n"
)
