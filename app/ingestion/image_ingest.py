from app.ingestion.schema import IngestedDocument

import pytesseract
from PIL import Image, ImageOps
import numpy as np

import os
from datetime import datetime

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\karth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# Load BLIP model 
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        print("DEBUG INPUT KEYS:", inputs.keys())

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption = processor.decode(out[0], skip_special_tokens=True)

        print("DEBUG CAPTION:", caption)

        return caption.strip()
    
    except Exception as e:
        print(f"BLIP ERROR: {e}")
        return ""

def ingest(file_path: str):
    try:
        image = Image.open(file_path)

        image = ImageOps.exif_transpose(image)
        
        image = image.convert("RGB")

        image = Image.fromarray(np.array(image))

        image.load()

    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")
    

    # Step 1: OCR text
    ocr_text = pytesseract.image_to_string(image).strip()

    # Step 2: Caption
    caption = generate_caption(image)

    # Step 3: Fallback 
    if not caption:
        caption = "An image (caption unavailable)"

    # Step 3: combine text
    final_text = ""

    if caption:
        final_text += f"Image Description: {caption}\n"
    
    if ocr_text:
        final_text += f"OCR Text: {ocr_text}"

    if not final_text.strip():
        raise ValueError("Image ingestion failed: No caption or OCR extracted")

    metadata = {
        "source": os.path.basename(file_path),
        "modality": "image",
        "caption": caption,
        "ingestion_time": datetime.utcnow().isoformat(),
        "ocr": True
    }

    return [
        IngestedDocument(
            text = final_text.strip(),
            metadata=metadata,
        )
    ]