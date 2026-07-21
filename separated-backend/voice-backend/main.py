import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

from model import load_model, predict

app = FastAPI(title="Voice Anti-Spoofing API")

origins = ["*"]  # tighten to your actual frontend URL once deployed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_REPO_ID = "LaabhGupta/voice-antispoofing"
MODEL_FILENAME = "baseline_cnn_finetuned.pth"
ARCHITECTURE = "baseline"
device = "cpu"

print("🔍 Downloading Voice model from Hugging Face Hub...")
weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)

print("🧠 Loading model...")
model = load_model(ARCHITECTURE, weights_path, device=device)
print("✔ Voice model loaded successfully")

ALLOWED_EXTENSIONS = ["wav", "mp3"]


@app.get("/")
def health():
    return {"message": "Voice Anti-Spoofing API Running!"}


@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"}

    try:
        file_bytes = await file.read()

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        try:
            label, confidence = predict(tmp.name, model, device=device)
        finally:
            os.remove(tmp.name)

        return {
            "filename": file.filename,
            "predicted_class": label,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": f"Failed to process audio: {e}"}
