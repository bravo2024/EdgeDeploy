from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["model_size_mb","latency_ms","throughput_qps","memory_usage_mb","cpu_util_pct","gpu_available","quantization_type","batch_size","input_size","framework_optimized"]
CATEGORICAL_FEATURES = ["quantization_type","gpu_available","framework_optimized"]
NUMERICAL_FEATURES = ["model_size_mb","latency_ms","throughput_qps","memory_usage_mb","cpu_util_pct","batch_size","input_size"]
TARGET_NAME = "deployment_success"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "model_size_mb": rng.lognormal(mean=2.5,sigma=1,size=n).clip(1,500).round(1),
        "latency_ms": rng.lognormal(mean=2,sigma=0.8,size=n).clip(0.5,100).round(1),
        "throughput_qps": rng.lognormal(mean=3,sigma=0.7,size=n).clip(1,500).round(1),
        "memory_usage_mb": rng.lognormal(mean=3.5,sigma=0.6,size=n).clip(10,2000).round(1),
        "cpu_util_pct": rng.uniform(10,95,size=n).round(1),
        "gpu_available": rng.choice([0,1],size=n,p=[0.3,0.7]),
        "quantization_type": rng.choice(["none","int8","fp16","fp32"],size=n,p=[0.15,0.30,0.35,0.20]),
        "batch_size": rng.choice([1,2,4,8,16,32],size=n,p=[0.05,0.10,0.20,0.30,0.25,0.10]),
        "input_size": rng.choice([224,256,384,512,640,1024],size=n,p=[0.15,0.20,0.25,0.20,0.10,0.10]),
        "framework_optimized": rng.choice([0,1],size=n,p=[0.4,0.6]),
    })
    size=np.clip(df["model_size_mb"]/500,0,1); lat=np.clip(df["latency_ms"]/100,0,1)
    thr=np.clip(df["throughput_qps"]/500,0,1); mem=np.clip(df["memory_usage_mb"]/2000,0,1)
    cpu=df["cpu_util_pct"]/100; gpu=df["gpu_available"]
    quant=df["quantization_type"].map({"none":0,"fp32":0.3,"fp16":0.5,"int8":1}).values
    batch=np.clip(df["batch_size"]/32,0,1); inp=np.clip(df["input_size"]/1024,0,1); opt=df["framework_optimized"]
    log_odds = 2.0 - 0.5*size - 0.6*lat + 0.5*thr - 0.4*mem - 0.3*cpu + 0.4*gpu + 0.5*quant + 0.2*batch - 0.2*inp + 0.4*opt + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,70)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(deployment_success=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
