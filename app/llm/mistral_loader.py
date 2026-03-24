from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )

    return model, tokenizer