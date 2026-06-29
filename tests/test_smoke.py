from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from src.data import make_synthetic, create_dataloaders
from src.model import build_mobilenet, evaluate_model

def test_data():
    data = make_synthetic(n=10, seed=42)
    assert data["images"].shape == (10, 3, 224, 224)
    assert data["n_samples"] == 10

def test_dataloader():
    data = make_synthetic(n=10, seed=42)
    tl, vl = create_dataloaders(data, batch_size=4)
    batch = next(iter(tl))
    assert batch[0].shape == (4, 3, 224, 224)

def test_model():
    model = build_mobilenet(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1000)

def test_evaluate():
    data = make_synthetic(n=20, seed=42)
    _, vl = create_dataloaders(data, batch_size=10)
    model = build_mobilenet(num_classes=1000)
    m = evaluate_model(model, vl, device="cpu")
    assert "accuracy" in m
    assert "avg_latency_ms" in m
    assert "throughput_fps" in m

def test_export_onnx():
    from src.model import export_to_onnx, get_model_size
    import os
    model = build_mobilenet(num_classes=1000)
    path = "models/test_model.onnx"
    export_to_onnx(model, path, opset_version=18)
    assert get_model_size(path) > 0
    if os.path.exists(path):
        os.remove(path)
