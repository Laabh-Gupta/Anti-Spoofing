import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import torch

from voice.model import load_model as load_voice_model, predict as predict_voice
from image.model import load_model as load_image_model, predict as predict_image
from text.extract_text import extract_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 1. Initialize FastAPI App and CORS ---
app = FastAPI(title="Multi-Modal Anti-Spoofing API")

origins = ["*"]  # tighten this to your actual unified frontend URL once deployed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

# --- 2. Load all three models at startup ---
print("🔍 Loading Voice model...")
voice_weights = hf_hub_download(repo_id="LaabhGupta/voice-antispoofing", filename="baseline_cnn_finetuned.pth")
voice_model = load_voice_model("baseline", voice_weights, device=device)
print("✔ Voice model loaded")

print("🔍 Loading Image model...")
image_weights = hf_hub_download(repo_id="LaabhGupta/image-antispoofing", filename="deeper_cnn_model.pth")
image_model = load_image_model("deeper", image_weights, device=device)
print("✔ Image model loaded")

import torch
import transformers
import packaging

print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("packaging version:", packaging.__version__)

from transformers.utils import is_torch_available
print("is_torch_available():", is_torch_available())

print("🔍 Loading Text model...")
text_model = AutoModelForSequenceClassification.from_pretrained("LaabhGupta/Text-Anti-Spoofing")

print("🔍 Loading Text model...")
text_model = AutoModelForSequenceClassification.from_pretrained("LaabhGupta/Text-Anti-Spoofing")
tokenizer = AutoTokenizer.from_pretrained("LaabhGupta/Text-Anti-Spoofing")
text_model.to(device)
text_model.eval()
print("✔ Text model loaded")

print("🎉 All models loaded - API ready")

# --- 3. Constants ---
TEXT_EXTENSIONS = ["docx", "pdf", "txt"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"]
VOICE_EXTENSIONS = ["wav", "mp3"]
TEXT_MAX_LENGTH = 256
TEXT_CLASS_NAMES = ["HUMAN", "AI"]


@app.get("/")
def health():
    return {"message": "Multi-Modal Anti-Spoofing API Running! (text, image, voice)"}


@app.post("/predict/text")
async def predict_text_endpoint(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in TEXT_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(TEXT_EXTENSIONS)}"}
    try:
        file_bytes = await file.read()
        text = extract_text(file.filename, file_bytes)
        if not text or len(text.strip()) < 20:
            return {"error": "Could not extract enough text from this file to analyze."}

        inputs = tokenizer(text, truncation=True, max_length=TEXT_MAX_LENGTH, return_tensors="pt",
                            return_token_type_ids=False).to(device)
        with torch.no_grad():
            logits = text_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()

        return {
            "filename": file.filename,
            "predicted_class": TEXT_CLASS_NAMES[idx],
            "confidence": float(probs[0][idx]),
        }
    except Exception as e:
        return {"error": f"Failed to process file: {e}"}


@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(IMAGE_EXTENSIONS)}"}
    try:
        file_bytes = await file.read()
        label, confidence = predict_image(file_bytes, image_model, device=device)
        return {"filename": file.filename, "predicted_class": label, "confidence": confidence}
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}


@app.post("/predict/voice")
async def predict_voice_endpoint(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in VOICE_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(VOICE_EXTENSIONS)}"}
    try:
        file_bytes = await file.read()
        import tempfile, os as os_module

        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(file_bytes)
        tmp.close()  # release the lock before torchaudio tries to open it

        try:
            label, confidence = predict_voice(tmp.name, voice_model, device=device)
        finally:
            os_module.remove(tmp.name)  # manual cleanup since delete=False skips auto-cleanup

        return {"filename": file.filename, "predicted_class": label, "confidence": confidence}
    except Exception as e:
        return {"error": f"Failed to process audio: {e}"}


@app.post("/predict/")
async def predict_auto(file: UploadFile = File(...)):
    """Single smart endpoint - detects file type and routes automatically. Ideal for the unified frontend."""
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext in TEXT_EXTENSIONS:
        return await predict_text_endpoint(file)
    elif ext in IMAGE_EXTENSIONS:
        return await predict_image_endpoint(file)
    elif ext in VOICE_EXTENSIONS:
        return await predict_voice_endpoint(file)
    else:
        return {"error": f"Unsupported file type: .{ext}"}