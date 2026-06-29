from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import argparse
import torch
from src.data import make_synthetic, create_dataloaders
from src.model import build_mobilenet, evaluate_model, export_to_onnx, benchmark_onnx, quantize_onnx, get_model_size
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = make_synthetic(n=a.n, seed=a.seed)
    _, val_loader = create_dataloaders(data, batch_size=a.batch_size, seed=a.seed)
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = build_mobilenet(num_classes=1000)
    print(f"MobileNet-V2 ({sum(p.numel() for p in model.parameters()):,} params)")

    results = {}

    print("Benchmarking PyTorch FP32...")
    pt_metrics = evaluate_model(model, val_loader, device=device)
    results["PyTorch FP32"] = {**pt_metrics, "size_mb": get_model_size("")}

    onnx_path = export_to_onnx(model)
    onnx_size = get_model_size(onnx_path)
    print(f"ONNX exported ({onnx_size:.2f} MB)")

    print("Benchmarking ONNX FP32...")
    onnx_metrics = benchmark_onnx(onnx_path, val_loader, num_threads=a.num_threads)
    results["ONNX FP32"] = {**onnx_metrics, "size_mb": onnx_size}

    print("Quantizing ONNX to INT8...")
    quant_path = quantize_onnx(onnx_path)
    quant_size = get_model_size(quant_path)
    print(f"Quantized ({quant_size:.2f} MB)")

    print("Benchmarking ONNX INT8...")
    quant_metrics = benchmark_onnx(quant_path, val_loader, num_threads=a.num_threads)
    results["ONNX INT8"] = {**quant_metrics, "size_mb": quant_size}

    print_report(results)
    save_metrics(results)
    print("Saved models/metrics.json, models/model.onnx, models/model_quantized.onnx")

if __name__ == "__main__":
    main()
