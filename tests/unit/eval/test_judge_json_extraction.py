"""The contract for app/eval/judges/qwen_judge.py::_extract_json_from_text.

One function serves two frameworks that want two different JSON shapes, and
they are not interchangeable:

  * Ragas prompts ask the judge for a top-level ARRAY
    (`[{"statement": ..., "verdict": 1}]`).
  * Every DeepEval schema is a top-level OBJECT wrapping a list —
    `Truths` {"truths": [...]}, `Claims` {"claims": [...]},
    `Verdicts` {"verdicts": [...]}.

The original implementation searched for an array BEFORE an object,
unconditionally, so every DeepEval reply was silently reduced to its inner
list and the wrapper object was thrown away. DeepEval's own parser
(`deepeval.metrics.utils.trimAndLoadJson`) is object-only — it locates the
payload with `find("{")` / `rfind("}")`, finds neither in a bare array, and
parses the empty string — so the mangled value came back as
"Evaluation LLM outputted an invalid JSON. Please use a better evaluation
model."

That shipped. In v1.0.0-rc3's first completed quality run
(quality-reports/deepeval/20260827-065236-live.json) 5 of DeepEval's 6
metrics reported `mean=None, n=0`, every row carrying that error string, and
the sixth scored exactly one row — the one whose reply happened to arrive in
a markdown code fence, which is the single branch that returned the object
intact. The resulting badge read "deepeval avg: 1.00" off that lone row.

Both directions are pinned here: DeepEval's object shape must survive, and
Ragas's array shape must not regress while fixing it.
"""

from __future__ import annotations

import json

from app.eval.judges.qwen_judge import _extract_json_from_text


def _roundtrip(raw: str):
    """Extract, then parse — what both framework adapters actually do."""
    return json.loads(_extract_json_from_text(raw))


class TestDeepEvalObjectShapeSurvives:
    """Object-wrapping-a-list must come back as the OBJECT, never the list."""

    def test_bare_object_with_list_value(self):
        raw = '{"truths": ["Revenue was $94.9B.", "Gross margin was 46.2%."]}'
        assert _roundtrip(raw) == {"truths": ["Revenue was $94.9B.", "Gross margin was 46.2%."]}

    def test_object_behind_a_prose_preamble(self):
        raw = 'Here is the evaluation:\n{"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]}'
        assert _roundtrip(raw) == {"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]}

    def test_object_in_a_json_code_fence(self):
        raw = '```json\n{"claims": ["Revenue grew year over year."]}\n```'
        assert _roundtrip(raw) == {"claims": ["Revenue grew year over year."]}

    def test_nested_object_is_not_truncated_at_first_inner_brace(self):
        """The old code-fence branch used a non-greedy `\\{.*?\\}` and cut a
        nested object at its first inner `}`."""
        raw = '```json\n{"verdicts": [{"verdict": "yes", "meta": {"n": 1}}]}\n```'
        assert _roundtrip(raw) == {"verdicts": [{"verdict": "yes", "meta": {"n": 1}}]}

    def test_extracted_object_is_parseable_by_deepevals_object_only_scan(self):
        """Mirrors trimAndLoadJson's actual locate step, which is why the
        wrapper object mattering is not a stylistic preference."""
        raw = '{"truths": ["a", "b"]}'
        extracted = _extract_json_from_text(raw)
        start = extracted.find("{")
        end = extracted.rfind("}") + 1
        assert start != -1 and end != 0, "deepeval could not locate an object"
        assert json.loads(extracted[start:end]) == {"truths": ["a", "b"]}


class TestRagasArrayShapeUnchanged:
    """The array-shaped replies Ragas asks for must keep working."""

    def test_top_level_array_of_objects(self):
        raw = '[{"statement": "s", "reason": "r", "verdict": 1}]'
        assert _roundtrip(raw) == [{"statement": "s", "reason": "r", "verdict": 1}]

    def test_array_behind_a_prose_preamble(self):
        raw = 'Sure, here you go:\n[{"statement": "x", "attributed": 1}]'
        assert _roundtrip(raw) == [{"statement": "x", "attributed": 1}]

    def test_array_in_a_code_fence(self):
        raw = '```json\n[{"sentence_index": 0, "simpler_statements": ["a"]}]\n```'
        assert _roundtrip(raw) == [{"sentence_index": 0, "simpler_statements": ["a"]}]

    def test_ragas_flat_object_reply(self):
        """Ragas's question-generation step wants an object, not an array."""
        raw = '{"question": "What was revenue?", "noncommittal": 0}'
        assert _roundtrip(raw) == {"question": "What was revenue?", "noncommittal": 0}


class TestPositionDecidesShape:
    """Whichever delimiter opens first wins — that is the whole rule."""

    def test_object_first_wins(self):
        assert _roundtrip('{"a": 1} then [2, 3]') == {"a": 1}

    def test_array_first_wins(self):
        assert _roundtrip('[2, 3] then {"a": 1}') == [2, 3]

    def test_falls_through_to_the_other_opener_when_the_first_is_prose(self):
        """A stray `{` in prose must not block a real array further along."""
        raw = 'Use the {placeholder} form. [{"verdict": 1}]'
        assert _roundtrip(raw) == [{"verdict": 1}]


class TestDegradedInput:
    def test_non_json_is_returned_as_is(self):
        assert _extract_json_from_text("no json here at all") == "no json here at all"

    def test_worker_error_envelope_survives(self):
        """generate() returns `{"error": ...}` on any failure path — that must
        stay intact so the caller reports the real reason."""
        assert _roundtrip('{"error": "worker died"}') == {"error": "worker died"}
