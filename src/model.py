from __future__ import annotations
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time

def build_mobilenet(num_classes=1000):
    import torchvision
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    return model

@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    latencies = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        start = time.perf_counter()
        outputs = model(images)
        latencies.append((time.perf_counter() - start) * 1000)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {
        "accuracy": correct / total,
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": float(1000 / np.mean(latencies) * loader.batch_size),
    }

def export_to_onnx(model, path="models/model.onnx", opset_version=18):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, x, path,
                      input_names=["input"], output_names=["output"],
                      opset_version=opset_version, dynamo=False)
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    return path

def benchmark_onnx(path, loader, num_threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    latencies = []
    for images, _ in loader:
        batch = images.numpy()
        start = time.perf_counter()
        sess.run(None, {input_name: batch})
        latencies.append((time.perf_counter() - start) * 1000)
    bs = loader.batch_size
    return {
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": float(1000 / np.mean(latencies) * bs),
        "num_threads": num_threads,
    }

def quantize_onnx(onnx_path, output_path="models/model_quantized.onnx"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import onnxruntime.quantization as quant
    quant.quantize_dynamic(onnx_path, output_path, weight_type=quant.QuantType.QInt8)
    return output_path

def get_model_size(path):
    return Path(path).stat().st_size / (1024 * 1024)
