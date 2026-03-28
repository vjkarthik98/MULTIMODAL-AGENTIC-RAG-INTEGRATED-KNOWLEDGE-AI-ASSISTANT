import os

def detect_modality(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt"]:
        return "text"
    elif ext in [".pdf", ".docx", ".xlsx", ".xls"]:
        return "document"
    elif ext in [".jpg", ".jpeg", ".png"]:
        return "image"
    elif ext in [".mp3", ".wav"]:
        return "audio"
    elif ext in [".mp4", ".avi"]:
        return "video"
    else:
        raise ValueError(f"Unsupported file type: {ext}")