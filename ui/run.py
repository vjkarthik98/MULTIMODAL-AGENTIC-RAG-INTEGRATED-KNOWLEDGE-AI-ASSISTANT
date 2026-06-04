"""
Launch the Gradio UI: python ui/run.py
Requires the FastAPI backend to be running (default: http://localhost:8000).
Set BACKEND_URL env var to override.
"""
import os
from ui.gradio_app import app
from ui.theme import CSS, SHOW_PASSWORD_JS

if __name__ == "__main__":
    app.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
        css=CSS,
        js=SHOW_PASSWORD_JS,
    )
