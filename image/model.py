"""
Model architectures for Image Anti-Spoofing System.

Detects whether an image is REAL or FAKE (AI-generated OR deepfake/manipulated).

Three architectures were trained and compared on prithivMLmods/AI-vs-Deepfake-vs-Real
(Artificial + Deepfake merged into one FAKE class vs REAL):

- BaselineCNN   - shallow 3-block CNN
- DeeperCNN     - deeper 4-block CNN with BatchNorm
- ViTModel      - ViT-B/16 (ImageNet pretrained) adapted for RGB images via transfer learning

Preprocessing pipeline (must match at inference time):
    image -> resize to 224x224 -> normalize with ImageNet mean/std
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

IMAGE_SIZE = 224
CLASS_NAMES = ["REAL", "FAKE"]  # matches training: 0=REAL, 1=FAKE (Artificial+Deepfake merged)

_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.linear_stack(self.flatten(self.conv_stack(x)))


class DeeperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(128 * 14 * 14, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.linear_stack(self.flatten(self.conv_stack(x)))


class ViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = models.vit_b_16(weights=None)  # weights loaded from state_dict, not ImageNet, at inference time
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


def preprocess_image(file_path_or_bytes):
    """Load an image (path or raw bytes) and convert it into the normalized tensor the models expect.
    Returns a tensor of shape (1, 3, 224, 224), ready to feed to any of the three models.
    """
    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        import io
        image = Image.open(io.BytesIO(file_path_or_bytes))
    else:
        image = Image.open(file_path_or_bytes)

    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = _transform(image)
    return tensor.unsqueeze(0)  # add batch dimension


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


def predict(file_path_or_bytes, model, device="cpu"):
    """Run inference on a single image. Returns (predicted_label, confidence)."""
    tensor = preprocess_image(file_path_or_bytes).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], float(probs[pred_idx])
