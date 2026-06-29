from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import streamlit as st
import pandas as pd
from src.model import build_mobilenet, evaluate_model, export_to_onnx, benchmark_onnx, quantize_onnx, get_model_size
from src.data import make_synthetic, create_dataloaders
from src.visualizations import plot_latency_comparison, plot_size_comparison, plot_throughput_comparison

st.set_page_config(page_title="EdgeDeploy | ONNX Edge Benchmark", layout="wide", page_icon="\U0001f4e1")

with st.sidebar:
    st.header("\u2699 Config")
    n = st.slider("Samples", 100, 2000, 500, 100)
    batch_size = st.slider("Batch Size", 1, 128, 32, 8)
    num_threads = st.slider("CPU Threads", 1, 8, 2, 1)
    run_benchmark = st.button("Run Benchmark")
    st.caption("EdgeDeploy | MobileNet + ONNX Runtime")

if run_benchmark:
    with st.spinner("Running benchmark..."):
        data = make_synthetic(n=n, seed=42)
        _, val_loader = create_dataloaders(data, batch_size=batch_size, seed=42)
        model = build_mobilenet(num_classes=1000)
        results = {}
        st.info("Benchmarking PyTorch FP32...")
        pt = evaluate_model(model, val_loader)
        results["PyTorch FP32"] = {**pt, "size_mb": 0}
        st.info("Exporting to ONNX...")
        onnx_path = export_to_onnx(model)
        onnx_size = get_model_size(onnx_path)
        st.info("Benchmarking ONNX FP32...")
        onnx = benchmark_onnx(onnx_path, val_loader, num_threads)
        results["ONNX FP32"] = {**onnx, "size_mb": onnx_size}
        st.info("Quantizing ONNX INT8...")
        quant_path = quantize_onnx(onnx_path)
        quant_size = get_model_size(quant_path)
        st.info("Benchmarking ONNX INT8...")
        quant = benchmark_onnx(quant_path, val_loader, num_threads)
        results["ONNX INT8"] = {**quant, "size_mb": quant_size}
        st.success("Benchmark complete!")
        st.session_state.results = results

st.title("EdgeDeploy: Edge AI Benchmark")

if "results" in st.session_state:
    results = st.session_state.results
    c1, c2, c3 = st.columns(3)
    c1.metric("PyTorch FP32", f'{results["PyTorch FP32"]["avg_latency_ms"]:.1f} ms',
              help=f'Throughput: {results["PyTorch FP32"]["throughput_fps"]:.0f} fps')
    c2.metric("ONNX FP32", f'{results["ONNX FP32"]["avg_latency_ms"]:.1f} ms',
              help=f'Throughput: {results["ONNX FP32"]["throughput_fps"]:.0f} fps')
    c3.metric("ONNX INT8", f'{results["ONNX INT8"]["avg_latency_ms"]:.1f} ms',
              help=f'Throughput: {results["ONNX INT8"]["throughput_fps"]:.0f} fps')

    rows = []
    for variant, r in results.items():
        rows.append({
            "Variant": variant,
            "Avg Latency (ms)": f'{r["avg_latency_ms"]:.2f}',
            "P50 (ms)": f'{r["p50_latency_ms"]:.2f}',
            "P95 (ms)": f'{r["p95_latency_ms"]:.2f}',
            "Throughput (fps)": f'{r["throughput_fps"]:.0f}',
            "Size (MB)": f'{r["size_mb"]:.2f}' if r["size_mb"] > 0 else "N/A",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Variant"), use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a: st.pyplot(plot_latency_comparison(results))
    with col_b: st.pyplot(plot_size_comparison(results))
    with col_c: st.pyplot(plot_throughput_comparison(results))
else:
    st.info("Click 'Run Benchmark' in the sidebar to start.")
    st.markdown("""
    **EdgeDeploy** benchmarks MobileNet-V2 across three deployment variants:
    - **PyTorch FP32**: Full precision, eager mode
    - **ONNX FP32**: ONNX Runtime with graph optimizations
    - **ONNX INT8**: Dynamic quantization (QInt8 weights)
    """)
