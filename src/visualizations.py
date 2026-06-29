from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#1a1f2e", "figure.facecolor": "#1a1f2e",
        "axes.edgecolor": "#4a5568", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "text.color": "white", "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": "#4a5568",
    })

def plot_latency_comparison(results):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = list(results.keys())
    latencies = [results[v]["avg_latency_ms"] for v in variants]
    colors = ["#22d3ee", "#a78bfa", "#f97316"]
    ax.bar(variants, latencies, color=colors[:len(variants)])
    ax.set_ylabel("Avg Latency (ms)", color="white")
    ax.set_title("Inference Latency by Deployment Variant", color="white")
    ax.grid(True, alpha=.2)
    for i, (v, l) in enumerate(zip(variants, latencies)):
        ax.text(i, l + 0.5, f"{l:.1f}ms", ha="center", color="white")
    return fig

def plot_size_comparison(results):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = list(results.keys())
    sizes = [results[v].get("size_mb", 0) for v in variants]
    colors = ["#22d3ee", "#a78bfa", "#f97316"]
    ax.bar(variants, sizes, color=colors[:len(variants)])
    ax.set_ylabel("Model Size (MB)", color="white")
    ax.set_title("Model Size Comparison", color="white")
    ax.grid(True, alpha=.2)
    for i, (v, s) in enumerate(zip(variants, sizes)):
        ax.text(i, s + 0.5, f"{s:.2f}MB", ha="center", color="white")
    return fig

def plot_throughput_comparison(results):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = list(results.keys())
    fps = [results[v]["throughput_fps"] for v in variants]
    colors = ["#22d3ee", "#a78bfa", "#f97316"]
    ax.bar(variants, fps, color=colors[:len(variants)])
    ax.set_ylabel("Throughput (fps)", color="white")
    ax.set_title("Inference Throughput", color="white")
    ax.grid(True, alpha=.2)
    for i, (v, f) in enumerate(zip(variants, fps)):
        ax.text(i, f + 0.5, f"{f:.0f} fps", ha="center", color="white")
    return fig
