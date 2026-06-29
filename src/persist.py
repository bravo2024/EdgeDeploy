from __future__ import annotations
import torch
from pathlib import Path

def save_model(model, path="models/model.pth"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model_class, path="models/model.pth", num_classes=1000):
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
