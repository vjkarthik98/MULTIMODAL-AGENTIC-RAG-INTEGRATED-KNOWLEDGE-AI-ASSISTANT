import os
import pytest

from app.pipeline.ingestion_pipeline import process_file

def test_document_pipeline_excel(tmp_path):
    import pandas as pd

    file_path = tmp_path/ "pipeline_text.xlsx"

    df = pd.DataFrame({
        "A": [1,2],
        "B": [3, 4]
    })

    df.to_excel(file_path, index=False)

    result = process_file(str(file_path), session_id="pipeline_test")

    assert result["status"] == "success"
    assert result["chunks"] > 0
    assert result["details"]["embedding_done"] is True

