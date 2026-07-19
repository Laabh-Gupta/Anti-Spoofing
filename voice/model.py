"""
Model architectures for Voice Anti-Spoofing System.

Three architectures were trained and compared:
- BaselineCNN   - shallow 3-block CNN over mel spectrograms
- DeeperCNN     - deeper 4-block CNN with BatchNorm (had training instability - see README)
- ViTModel      - ViT-B/16 (ImageNet pretrained) adapted for 1-channel spectrogram input via transfer learning

Preprocessing pipeline (must match at inference time):
    raw audio -> resample to 16kHz -> mono -> pad/trim to 4 seconds
    -> MelSpectrogram(n_fft=1024, hop_length=512, n_mels=128)
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import torchvision.transforms as TV

# --- Config (must match training) ---
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SECS = 4
TARGET_LEN = TARGET_SAMPLE_RATE * TARGET_LEN_SECS
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

CLASS_NAMES = ["fake", "real"]  # matches sorted() order used during training - verify against your dataset folder names


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(15360, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.linear_stack(self.flatten(self.conv_stack(x)))


class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(128 * 8 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.linear_stack(self.flatten(self.conv_stack(x)))


class ViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        original_conv = self.vit.conv_proj
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        self.vit.conv_proj.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        self.resizer = TV.Resize((224, 224), antialias=True)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(self.resizer(x))


# --- Preprocessing ---
_mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)


def preprocess_audio(file_path):
    """Load an audio file and convert it into the mel spectrogram tensor the models expect.
    Returns a tensor of shape (1, 1, N_MELS, time_frames), ready to feed to any of the three models.
    """
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if waveform.shape[1] > TARGET_LEN:
        waveform = waveform[:, :TARGET_LEN]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, TARGET_LEN - waveform.shape[1]))

    spectrogram = _mel_spectrogram_transform(waveform)
    return spectrogram.unsqueeze(0)  # add batch dimension


def load_model(architecture, weights_path, device="cpu"):
    """architecture: one of 'baseline', 'deeper', 'vit'"""
    archs = {"baseline": BaselineCNN, "deeper": DeeperCNN, "vit": ViTModel}
    if architecture not in archs:
        raise ValueError(f"architecture must be one of {list(archs.keys())}")
    model = archs[architecture]()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(file_path, model, device="cpu"):
    """Run inference on a single audio file. Returns (predicted_label, confidence)."""
    spectrogram = preprocess_audio(file_path).to(device)
    with torch.no_grad():
        logits = model(spectrogram)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], float(probs[pred_idx])
