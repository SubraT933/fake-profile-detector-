# Social Profile Fraud and Bot Detection Framework

A modular, production-ready machine learning framework for detecting synthetic, automated, and fraudulent social media accounts. The system combines topological Graph Neural Networks (Graph Attention Networks and Graph Convolutional Networks) with traditional and ensemble classifiers to evaluate account integrity, trace local community motifs, and deliver local model explainability.

---

## 1. Architecture and Methodology

The system models social media profiles as nodes embedded in a continuous feature metric space. While graph neural networks traditionally require explicit social follower graphs (which are frequently unavailable due to platform privacy barriers and API rate limits), this architecture constructs a $k$-nearest-neighbor feature-similarity graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ directly from profile behavioral attributes.

### System Pipeline

```
[ Raw Profile Input / Link ]
            │
            ▼
[ Feature Extractor & Preprocessor ] ──► (Numeric Validation & Standardization)
            │
            ▼
[ k-NN Manifold Construction ] ───────► (Sparse Symmetric Adjacency & Edge Index)
            │
            ▼
[ Graph Neural Network Layer ] ───────► (GAT Parameterized Attention / GCN Propagation)
            │
            ├──────────────────────────────────────────┐
            ▼                                          ▼
[ Inductive Out-of-Graph Inference ]     [ Subgraph Mask Optimization ]
  - Inverse-Distance Aggregation           - GNNExplainer Mutual Information
  - Synthetic Risk Probability             - Saliency Feature Attribution
  - Community Motif Classification         - Edge Influence Decomposition
```

### Key Components

1. **Topology Construction**: Symmetric $k$-nearest-neighbor graph computed via Euclidean distance across scaled continuous features. Normalized Laplacian degree scaling is computed for GCN, and directed edge indices are constructed for GAT.
2. **Graph Attention Network (GAT)**: Computes parameterized edge attention coefficients $\alpha_{ij} = \text{Softmax}_j\left(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_j])\right)$, dynamically learning which neighboring accounts represent actionable signals of coordinated bot behavior versus organic noise.
3. **Inductive Out-of-Graph Inference**: Unseen query accounts are projected onto the learned manifold using fitted parameter scalers. Predictions are aggregated from the $k$-nearest manifold anchors via inverse-distance weighting.
4. **Local Subgraph Explainability (GNNExplainer)**: Joint optimization of continuous feature masks $m_{\text{feat}} \in [0, 1]^d$ and edge masks $m_{\text{edge}} \in [0, 1]^k$ with $L_1$ sparsity and entropy regularization penalties.
5. **Topological Community Motif Discovery**: Neighborhood graph decomposition categorizing the account's local cluster into *Coordinated Synthetic Cluster*, *Organic Human Cluster*, or *Mixed Boundary Topology*.

---

## 2. Directory Structure

```
fake-profile-detector/
├── pyproject.toml              # Build specification, metadata, and pytest configuration
├── requirements.txt            # Pinned runtime dependencies
├── README.md                   # Technical documentation and system guide
├── app.py                      # Production Streamlit web platform with executive UI
├── run_benchmark.py            # CLI utility for multi-model & multi-dataset evaluation
├── benchmark_engine.py         # Compatibility entry point delegating to core package
├── ml_model.py                 # Backward-compatibility API facade
├── url_extractor.py            # Backward-compatibility feature extraction facade
├── datasets/                   # Verified benchmark datasets (CSV format)
│   ├── Cresci_2015.csv
│   ├── Fake_Instagram_Profile_Detection.csv
│   ├── Instafake_Automated_Behavior.csv
│   ├── Instafake_Fake_vs_Real.csv
│   ├── Instagram_5k.csv
│   ├── Instagram_Spammer_Genuine.csv
│   ├── LIMFADD.csv
│   ├── TwiBot_22.csv
│   └── user_custom_dataset.csv
├── reports/                    # Persisted evaluation benchmarks and ranking summaries
├── src/
│   └── fake_profile_detector/  # Core Python package
│       ├── __init__.py
│       ├── config.py           # Configuration classes, file paths, and hyperparameters
│       ├── data.py             # Schema detection, data cleaning, and preprocessing
│       ├── graph.py            # Topology construction and sparse matrix operations
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gnn.py          # PyTorch GAT and GCN implementations
│       │   └── baselines.py    # Traditional & ensemble classifiers (RF, GBDT, SVM, MLP)
│       ├── explainability.py   # GNNExplainer mask optimization and gradient saliency
│       ├── extractor.py        # Profile URL parser and heuristic feature extractor
│       ├── pipeline.py         # End-to-end training, inference, and motif analysis
│       └── benchmark.py        # Multi-model benchmarking suite
└── tests/                      # Automated unit test suite
    ├── __init__.py
    ├── test_data.py
    ├── test_extractor.py
    ├── test_graph.py
    ├── test_models.py
    └── test_pipeline.py
```

---

## 3. Installation and Setup

### Prerequisites

- Python 3.9+ (tested through Python 3.13)
- PyTorch 2.0+

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/SubraT933/fake-profile-detector-.git
cd fake-profile-detector-

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode with dependencies
pip install -e .
```

Alternatively, install dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 4. Usage

### Launching the Web Platform

Start the executive web dashboard:

```bash
streamlit run app.py
```

The web interface provides:
- **Account Verification**: Enter any Instagram URL or handle for automated feature extraction, synthetic risk scoring, topological neighborhood analysis, and feature attribution.
- **Model Benchmarks**: Interactive performance leaderboard comparing 11 classifier algorithms across key evaluation metrics with training convergence curves.
- **Dataset Profiles**: Summary distributions, feature definitions, and raw previews across all included datasets.
- **Technical Architecture**: In-depth mathematical formulation of graph attention mechanisms, loss objectives, and mask optimization.

### Running Command-Line Benchmarks

Run an evaluation benchmark on the default dataset (`Instagram_5k.csv`):

```bash
python run_benchmark.py --dataset Instagram_5k.csv
```

Run benchmarks across all available datasets in `datasets/`:

```bash
python run_benchmark.py --all-datasets
```

List all discovered datasets:

```bash
python run_benchmark.py --list-datasets
```

Configure training epochs and random seeds:

```bash
python run_benchmark.py --dataset LIMFADD.csv --epochs 50 --seed 42 --output-dir reports/
```

### Python API Integration

```python
import pandas as pd
from fake_profile_detector.config import ModelConfig
from fake_profile_detector.data import load_dataset
from fake_profile_detector.pipeline import train_model, predict_profile, analyze_community_motif

# 1. Load data and train Graph Attention Network
df = load_dataset("Instagram_5k.csv")
config = ModelConfig(model_type="GAT", k_neighbors=8, epochs=40)
artifact = train_model(df, config=config)

# 2. Evaluate an unseen profile
query_profile = {
    "profile pic": 1.0,
    "nums/length username": 0.35,
    "fullname words": 1.0,
    "nums/length fullname": 0.0,
    "name==username": 0.0,
    "description length": 12.0,
    "external URL": 0.0,
    "private": 0.0,
    "#posts": 3.0,
    "#followers": 42.0,
    "#follows": 1850.0,
    "followers_following_ratio": 0.0227,
}

result = predict_profile(artifact, query_profile)
print(f"Synthetic Risk: {result['fake_probability']:.1%}")
print(f"Classification: {'Synthetic / Automated' if result['is_fake'] else 'Authentic / Human'}")

# 3. Analyze topological community motif
motif = analyze_community_motif(artifact, result)
print(f"Community Category: {motif['cluster_category']}")
print(f"Weighted Bot Affinity: {motif['weighted_affinity']:.1%}")
```

---

## 5. Benchmark Methodology and Algorithms

The benchmark suite systematically evaluates four distinct model families across identical stratified holdout partitions:

| Model Family | Algorithms Evaluated | Primary Characteristics |
|---|---|---|
| **Graph Neural Networks** | Graph Attention Network (GAT), Graph Convolutional Network (GCN) | Captures feature manifold topology and dynamic attention-weighted edge interactions. |
| **Ensemble Learning** | Random Forest, HistGradientBoosting, AdaBoost, Soft Voting Ensemble | Tree ensembles with gradient boosting and bagging mechanisms. |
| **Linear & Probabilistic** | Regularized Logistic Regression, Linear SVC (Calibrated), Gaussian Naive Bayes | Fast parametric baselines establishing linear separability thresholds. |
| **Deep Learning** | Multi-Layer Perceptron (MLP) | Multi-layer feedforward architecture with early stopping and Adam optimization. |

### Evaluation Metrics

Every model is assessed using:
- **Accuracy** and **Balanced Accuracy** (handles class imbalance)
- **Precision**, **Recall**, and **Specificity**
- **F1-Score** (harmonic mean of precision and recall)
- **ROC-AUC** and **Precision-Recall AUC (PR-AUC)**
- **Training Latency** (wall-clock seconds)

---

## 6. Testing and Quality Assurance

The codebase includes an automated test suite verifying data loaders, topology builders, model forward passes, feature extraction, and end-to-end pipeline execution:

```bash
pytest
```

To run with verbose output and coverage reporting:

```bash
pytest -v --tb=short
```

---

## 7. License and Attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
