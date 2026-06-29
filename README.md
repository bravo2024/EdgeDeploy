# EdgeDeploy

> Edge AI deployment success prediction with TVM auto-tuning and hardware optimisation.

Trains four classifiers on synthetic edge deployment data (model size, latency, memory usage, throughput, quantization type, target hardware) to predict successful deployment. Dashboard covers performance benchmarking, TVM compilation theory, and hardware mapping exploration.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Random Forest) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.743 |
| Gini | 0.486 |
| KS Statistic | 0.436 |
| F1 Score | 0.517 |
| Accuracy | 0.656 |

5-fold CV AUC: 0.755 ± 0.058. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Deployment records overview, success/failure distribution, feature descriptions |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Performance** | Latency and memory distributions by deployment status, throughput vs model size scatter |
| **Deployment** | TVM auto-tuning theory, Amdahl's Law, quantization step size, deployment success by quantization type |

## Repo Structure

```
EdgeDeploy/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic edge deployment dataset: model size, latency, memory usage, throughput, quantization type, target hardware, batch size, and deployment success label.

## License

MIT
