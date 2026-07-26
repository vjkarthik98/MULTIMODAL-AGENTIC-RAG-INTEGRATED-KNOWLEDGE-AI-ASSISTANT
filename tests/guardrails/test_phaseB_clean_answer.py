"""Phase B — clean answer body: parse [N] for chips, strip all citations/markers
with no whitespace gaps, no references or filename leakage in prose."""
from app.core.response import strip_inline_citations, extract_cited_indices
from app.guardrails import output_guard


def test_numeric_citations_stripped_no_gaps():
    out = strip_inline_citations(
        "Net sales were $383.3 billion [1] in fiscal 2023 [2,3]."
    )
    assert "[" not in out and "]" not in out
    assert "  " not in out               # no double space (the "of  net sales" bug)
    assert " ." not in out               # no space before period
    assert out == "Net sales were $383.3 billion in fiscal 2023."


def test_structured_markers_stripped():
    out = strip_inline_citations(
        "Revenue came from [Sheet: Sales Data, Rows 1-5] and page [PG:3]."
    )
    assert "Sheet" not in out and "PG" not in out
    assert "[" not in out


def test_filename_citation_stripped():
    out = strip_inline_citations("The figure is shown in [gdp_report.pdf].")
    assert "gdp_report" not in out
    assert out == "The figure is shown in."


def test_timestamp_and_speaker_markers_stripped():
    out = strip_inline_citations("He said [T:12.0s] [SPK:Alice] hello.")
    assert "T:" not in out and "SPK" not in out


def test_doc_stem_and_pii_mangled_citation_stripped():
    # Live-observed leaks: the LLM cited a filename stem, and the PII scrubber
    # mangled ".docx" into "<URL>". Both must be stripped from the prose.
    out = strip_inline_citations(
        "Net sales were $394.3 billion [aapl_def14a_<URL>cx] and income $99.8 billion [aapl_def14a_2023]."
    )
    assert "aapl_def14a" not in out
    assert "<URL>" not in out and "[" not in out
    assert out == "Net sales were $394.3 billion and income $99.8 billion."


def test_extract_cited_indices_multi():
    assert extract_cited_indices("a [1] b [2,3] c [10]") == {1, 2, 3, 10}
    assert extract_cited_indices("no citations here") == set()


def test_output_guard_does_not_delete_citations():
    """Guard must leave [2,3] intact so extract_cited_indices can read it."""
    answer = "Both figures agree [2,3] and the total is $383.3 billion."
    res = output_guard.check(
        answer=answer,
        context_chunks=["total net sales were $383.3 billion in 2023"],
        sources=[{"filename": "a.pdf"}, {"filename": "b.pdf"}, {"filename": "c.pdf"}],
    )
    # Citations are NOT removed by the guard (downstream stripper owns that).
    assert "[2,3]" in res.text
    assert extract_cited_indices(res.text) == {2, 3}


def test_strip_after_guard_is_clean():
    """End-to-end Phase B contract: guard (non-destructive) then strip = clean."""
    answer = "Net sales of $383.3 billion [1] fell 3% [2,3]."
    res = output_guard.check(
        answer=answer,
        context_chunks=["net sales of $383.3 billion fell 3%"],
        sources=[{"filename": "a.pdf"}, {"filename": "b.pdf"}, {"filename": "c.pdf"}],
    )
    cited = extract_cited_indices(res.text)
    final = strip_inline_citations(res.text)
    assert cited == {1, 2, 3}
    assert "[" not in final and "  " not in final and " ." not in final
    assert final == "Net sales of $383.3 billion fell 3%."


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
