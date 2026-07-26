"""Phase C — number-grounding precision. Legit financial figures (rounded,
scaled, $/comma-formatted) must NOT be flagged; genuine fabrications must be."""
from app.eval.metrics.hallucination import _numbers_grounded, hallucination_flag_single


# --- The exact false positives seen in logs/app.log must now be grounded ----

def test_rounded_billions_match_raw_millions():
    # Answer rounds; 10-K table prints raw thousands/millions.
    ctx = ["Total net sales were $314,623 and net income was $84,289 (in millions)."]
    grounded, ung = _numbers_grounded(
        "Net sales were $314.6 billion and net income $84.3 billion.", ctx
    )
    assert grounded, f"should be grounded, got ungrounded={ung}"


def test_decreased_by_figure_match():
    ctx = ["Net sales decreased 3% or $11,529 million during fiscal 2023."]
    grounded, ung = _numbers_grounded("Sales fell 3%, about $11.5 billion.", ctx)
    assert grounded, ung


def test_year_not_flagged():
    grounded, ung = _numbers_grounded(
        "In 2023 the company grew.", ["The fiscal 2023 results were strong."]
    )
    assert grounded and ung == []


def test_sec_accession_id_not_flagged():
    # SEC accession/CIK ids must never be treated as a quantitative claim.
    grounded, ung = _numbers_grounded(
        "See filing 000121935523000039 (CIK 1219355).",
        ["Apple Inc. annual report."],
    )
    assert grounded and ung == [], ung


def test_genuine_fabrication_is_flagged():
    ctx = ["Total net sales were $314,623 million in fiscal 2023."]
    grounded, ung = _numbers_grounded(
        "Net sales were $999.9 billion in 2023.", ctx
    )
    assert not grounded
    assert any("999" in u for u in ung)


def test_full_flag_clears_for_correct_answer():
    res = hallucination_flag_single(
        "Apple's net sales were $383.3 billion and net income $97.0 billion.",
        ["Net sales $383,285 million; net income $96,995 million (in millions)."],
    )
    assert res["flagged"] is False, res


def test_full_flag_set_for_wrong_answer():
    res = hallucination_flag_single(
        "Apple's net sales were $500.0 billion.",
        ["Net sales $383,285 million (in millions)."],
    )
    assert res["flagged"] is True, res


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
