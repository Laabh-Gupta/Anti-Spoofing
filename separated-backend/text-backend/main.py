import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from extract_text import extract_text

app = FastAPI(title="Text Anti-Spoofing API")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_REPO_ID = "LaabhGupta/Text-Anti-Spoofing"
device = "cpu"

print("🔍 Downloading Text model from Hugging Face Hub...")
model = AutoModelForSequenceClassification.from_pretrained(
    HF_REPO_ID,
    low_cpu_mem_usage=True,   # avoids duplicating tensors during weight loading - cuts peak RAM
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model.to(device)
model.eval()

gc.collect()  # force-release any temporary memory used during loading

print("✔ Text model loaded successfully")

MAX_LENGTH = 256
CLASS_NAMES = ["HUMAN", "AI"]
ALLOWED_EXTENSIONS = ["docx", "pdf", "txt"]


@app.get("/")
def health():
    return {"message": "Text Anti-Spoofing API Running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"}

    try:
        file_bytes = await file.read()
        text = extract_text(file.filename, file_bytes)

        if not text or len(text.strip()) < 20:
            return {"error": "Could not extract enough text from this file to analyze."}

        inputs = tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()

        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[idx],
            "confidence": float(probs[0][idx]),
            "extracted_chars": len(text),
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to process file: {e}"}