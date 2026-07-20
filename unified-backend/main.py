import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import torch

from voice.model import load_model as load_voice_model, predict as predict_voice
from image.model import load_model as load_image_model, predict as predict_image
from text.extract_text import extract_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="Multi-Modal Anti-Spoofing API")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

# --- Lazy-loaded model caches - start empty, load on first use only ---
_voice_model = None
_image_model = None
_text_model = None
_tokenizer = None


def get_voice_model():
    global _voice_model
    if _voice_model is None:
        print("🔍 Loading Voice model (first use)...")
        weights = hf_hub_download(repo_id="LaabhGupta/voice-antispoofing", filename="baseline_cnn_finetuned.pth")
        _voice_model = load_voice_model("baseline", weights, device=device)
        print("✔ Voice model loaded")
    return _voice_model


def get_image_model():
    global _image_model
    if _image_model is None:
        print("🔍 Loading Image model (first use)...")
        weights = hf_hub_download(repo_id="LaabhGupta/image-antispoofing", filename="deeper_cnn_model.pth")
        _image_model = load_image_model("deeper", weights, device=device)
        print("✔ Image model loaded")
    return _image_model


def get_text_model():
    global _text_model, _tokenizer
    if _text_model is None:
        print("🔍 Loading Text model (first use)...")
        _text_model = AutoModelForSequenceClassification.from_pretrained("LaabhGupta/Text-Anti-Spoofing")
        _tokenizer = AutoTokenizer.from_pretrained("LaabhGupta/Text-Anti-Spoofing")
        _text_model.to(device)
        _text_model.eval()
        print("✔ Text model loaded")
    return _text_model, _tokenizer


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
        text_model, tokenizer = get_text_model()
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
        model = get_image_model()
        file_bytes = await file.read()
        label, confidence = predict_image(file_bytes, model, device=device)
        return {"filename": file.filename, "predicted_class": label, "confidence": confidence}
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}


@app.post("/predict/voice")
async def predict_voice_endpoint(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in VOICE_EXTENSIONS:
        return {"error": f"Please upload one of: {', '.join(VOICE_EXTENSIONS)}"}
    try:
        model = get_voice_model()
        file_bytes = await file.read()
        import tempfile, os as os_module
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(file_bytes)
        tmp.close()
        try:
            label, confidence = predict_voice(tmp.name, model, device=device)
        finally:
            os_module.remove(tmp.name)
        return {"filename": file.filename, "predicted_class": label, "confidence": confidence}
    except Exception as e:
        return {"error": f"Failed to process audio: {e}"}


@app.post("/predict/")
async def predict_auto(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext in TEXT_EXTENSIONS:
        return await predict_text_endpoint(file)
    elif ext in IMAGE_EXTENSIONS:
        return await predict_image_endpoint(file)
    elif ext in VOICE_EXTENSIONS:
        return await predict_voice_endpoint(file)
    else:
        return {"error": f"Unsupported file type: .{ext}"}