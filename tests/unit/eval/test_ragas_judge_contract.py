"""The contract between app/eval/judges/qwen_judge.py and ragas 0.1.21.

Two defects are pinned here. Both were found by reading the ragas wheel's own
source rather than by observing a failure, because the run that exhibited them
(v1.0.0-rc5, CD run 33149188726) took its self-hosted runner down with it and
never uploaded a log.

1. **LLMResult generation shape.** `AnswerRelevancy._ascore`
   (ragas/metrics/_answer_relevance.py) asks for `n=strictness` generations and
   then reads them out of `result.generations[0]`:

       result = await self.llm.generate(prompt, n=self.strictness, ...)
       answers = [
           await _output_parser.aparse(result.text, prompt, self.llm)
           for result in result.generations[0]
       ]

   so the n generations must live in ONE inner list — `[[g1, g2, g3]]`. The
   wrapper built `[[g1], [g2], [g3]]` instead, one generation per outer entry,
   so `generations[0]` held a single item and `strictness` was silently always
   1 no matter what ragas asked for.

2. **The repair prompt was never recognised.** When a reply fails to parse,
   `RagasoutputParser.aparse` (ragas/llms/output_parser.py) re-prompts with
   `FIX_OUTPUT_FORMAT`, whose instruction is "Below, the Completion did not
   satisfy the constraints given in the Prompt." — and whose rendered text
   EMBEDS the entire original prompt as an input key. Every branch of
   `_build_ragas_prompt` therefore matched that embedded copy and told the
   judge to answer the original question again, reproducing the same reply
   that had just failed to parse and burning the single retry ragas allows.
   The repair prompt must be detected before any content branch runs.

Neither test imports ragas: the shapes and strings under test are ours, and
the repo's CI does not install the eval extras for unit runs.
"""

from __future__ import annotations

import pytest

from app.eval.judges import qwen_judge

# The exact instruction string from ragas/llms/output_parser.py's
# FIX_OUTPUT_FORMAT prompt. If a ragas upgrade changes this wording, the
# repair-prompt branch stops firing silently — this constant is the canary.
RAGAS_FIX_INSTRUCTION = (
    "Below, the Completion did not satisfy the constraints given in the Prompt."
)

# Abridged but structurally faithful: ragas' real context_recall prompt, as it
# appears INSIDE the repair prompt's `prompt` input key.
CONTEXT_RECALL_PROMPT = (
    "Given a context, and an answer, analyze each sentence in the answer and "
    'classify if the sentence can be attributed to the given context or not. Use only "Yes" '
    '(1) or "No" (0) as a binary classification. Output json with reason.'
)


class _StubJudge(qwen_judge.QwenRagasJudge):
    """QwenRagasJudge with the model call replaced by a fixed reply.

    Subclassing rather than monkeypatching `generate` keeps the test away from
    the worker subprocess entirely — `_run` is the only seam that touches it.
    """

    reply = '[{"statement": "s", "reason": "r", "attributed": 1}]'

    def _run(self, prompt_text: str) -> str:  # type: ignore[override]
        return self.reply


def test_generations_are_one_inner_list_of_n() -> None:
    """ragas reads generations[0] and expects n items in it."""
    judge = _StubJudge()
    result = judge.generate_text("any prompt", n=3)

    assert len(result.generations) == 1, (
        "n generations must share ONE inner list; ragas only ever reads "
        "generations[0] and would see a single item otherwise"
    )
    assert len(result.generations[0]) == 3
    assert all(g.text == _StubJudge.reply for g in result.generations[0])


def test_generations_default_n_is_single() -> None:
    judge = _StubJudge()
    result = judge.generate_text("any prompt")
    assert len(result.generations) == 1
    assert len(result.generations[0]) == 1


@pytest.mark.asyncio
async def test_async_generations_match_sync_shape() -> None:
    """agenerate_text is the path ragas actually takes.

    BaseRagasLLM.generate defaults to is_async=True, so the async wrapper — not
    the sync one — is what runs under `ragas.evaluate()`. It carried an
    identical copy of the shape bug.
    """
    judge = _StubJudge()
    result = await judge.agenerate_text("any prompt", n=3)
    assert len(result.generations) == 1
    assert len(result.generations[0]) == 3


def test_repair_prompt_is_detected_before_content_branches() -> None:
    """The repair prompt embeds the original, and must still be recognised."""
    repair_prompt = (
        f"{RAGAS_FIX_INSTRUCTION}\n"
        f"prompt: {CONTEXT_RECALL_PROMPT}\n"
        'completion: {"oops": not json\n'
        "fixed_completion: "
    )
    system = qwen_judge._build_ragas_prompt(repair_prompt)

    assert "corrected" in system.lower(), (
        "a repair prompt must ask for a CORRECTION of the failed completion, "
        "not a fresh answer to the embedded original question"
    )
    # The give-away that a content branch won: it names that branch's schema.
    assert '"attributed"' not in system


def test_content_branches_still_win_when_there_is_no_repair_wrapper() -> None:
    """The repair branch must not swallow ordinary prompts."""
    system = qwen_judge._build_ragas_prompt(CONTEXT_RECALL_PROMPT)
    assert '"attributed"' in system
    assert "corrected" not in system.lower()


def test_system_prompt_does_not_forbid_the_fence_ragas_asks_for() -> None:
    """ragas' own user prompt demands a fenced reply; we must not ban one.

    JSON_FORMAT_INSTRUCTIONS (ragas/llms/output_parser.py) ends with "return
    only a pure JSON string surrounded by triple backticks (```)". The system
    prompt used to say "No markdown", handing the judge two incompatible orders
    in a single turn. `_extract_json_from_text` unwraps a fence in its first
    branch, so a fenced reply is the cheapest one to parse — there was never a
    reason to forbid it.
    """
    for prompt in (CONTEXT_RECALL_PROMPT, "Generate a question ... noncommittal"):
        system = qwen_judge._build_ragas_prompt(prompt)
        assert "no markdown" not in system.lower()


def test_fenced_reply_survives_extraction() -> None:
    """The other half of the same contract, end to end."""
    fenced = '```json\n[{"statement": "s", "reason": "r", "attributed": 1}]\n```'
    assert qwen_judge._extract_json_from_text(fenced) == (
        '[{"statement": "s", "reason": "r", "attributed": 1}]'
    )
