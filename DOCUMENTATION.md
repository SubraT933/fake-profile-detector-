# 🛡️ Fake Instagram Profile Detector — Comprehensive Technical Documentation
*Enhanced with Saez Lab's SpaCEy Graph Attention & Topological Explainability*

This document provides a complete technical specification of the Fake Instagram Profile Detection system, covering the dataset, feature engineering, graph construction, deep learning architectures (GAT and GCN), training workflow with learning rate scheduling, out-of-graph inference, dual explainability engines, and the Streamlit web application.

---

## 📑 Table of Contents
1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Dataset Details & Statistical Profile](#2-dataset-details--statistical-profile)
3. [Feature Engineering & Preprocessing](#3-feature-engineering--preprocessing)
4. [Graph Construction: Feature-Similarity Topology](#4-graph-construction-feature-similarity-topology)
5. [Deep Learning Architectures: GAT vs GCN](#5-deep-learning-architectures-gat-vs-gcn)
6. [Training & Adaptive Optimization Pipeline](#6-training--adaptive-optimization-pipeline)
7. [Inference Engine for New User Inputs](#7-inference-engine-for-new-user-inputs)
8. [Dual Explainability Suite (GNNExplainer & Saliency)](#8-dual-explainability-suite-gnnexplainer--saliency)
9. [Topological Motif & Bot Syndicate Discovery](#9-topological-motif--bot-syndicate-discovery)
10. [Web Application Structure & User Interface](#10-web-application-structure--user-interface)
11. [How to Run & Configure](#11-how-to-run--configure)

---

## 1. Project Overview & Architecture

The application is an end-to-end Machine Learning web application designed to classify Instagram user accounts as either **Real (0)** or **Fake (1)**.

### Architectural Blueprint

```
+---------------------------------------------------------------------------------+
|                                Streamlit Frontend (app.py)                     |
|                                                                                 |
|   +-----------------------+   +------------------------+   +----------------+   |
|   | 🔍 Check a Profile   |   | 📈 Model Performance   |   | 📂 Dataset Tab |   |
|   +-----------------------+   +------------------------+   +----------------+   |
|   | 🧬 SpaCEy Methods & Research Tab (Saez Lab Nature Communications Guide)     |   |
+---------------------------------------+-----------------------------------------+
                                        |
               Calls ML Pipeline Methods|
                                        v
+---------------------------------------------------------------------------------+
|                                ML Core (ml_model.py)                           |
|                                                                                 |
| 1. prepare_data()            -> Imputes NaNs, encodes labels, extracts features |
| 2. build_graph()             -> Constructs k-NN feature graph, normalized A, &  |
|                                 edge_index tensor for Graph Attention           |
| 3. GAT & GCN Modules         -> SpaCEy-inspired Graph Attention Network (GAT)   |
|                                 and Kipf-Welling Baseline (GCN)                 |
| 4. train_gnn()               -> Adam optimizer + ReduceLROnPlateau scheduler    |
| 5. predict_user_profile()    -> Distance-weighted k-NN neighbor aggregation     |
| 6. GNNExplainer              -> Joint feature & edge mask optimization with     |
|                                 mutual information and entropy penalties        |
| 7. analyze_subgraph_motif()  -> Bot Syndicate vs Authentic Community detection  |
+---------------------------------------------------------------------------------+
                                        ^
                                        | Loads Data
+---------------------------------------+-----------------------------------------+
|                              Dataset (Instagram.csv)                            |
|                          5,000 labelled user records                            |
+---------------------------------------------------------------------------------+
```

---

## 2. Dataset Details & Statistical Profile

The model is trained on `Instagram.csv`, containing **5,000 labelled user profiles** with an exact 50/50 balance:
- **Total Samples**: 5,000
- **Real Profiles (`fake = 0`)**: 2,500 (50.0%)
- **Fake Profiles (`fake = 1`)**: 2,500 (50.0%)
- **Input Features**: 12 tabular columns
- **Target Variable**: `fake` (binary integer: 0 or 1)

### All 12 Input Features Explained

| # | Feature Name | Data Type | Description | Behavioral Intuition |
|---|---|---|---|---|
| 1 | `profile pic` | Binary (0 / 1) | Whether the account has an avatar/profile picture (1 = Yes, 0 = No). | Fake accounts frequently lack custom profile pictures. |
| 2 | `nums/length username` | Continuous [0, 1] | Ratio of numeric digits to username length (e.g., `user184920` has high ratio). | Automated bot generators commonly append random digits to usernames. |
| 3 | `fullname words` | Integer (≥ 0) | Count of whitespace-separated words in displayed full name. | Bots often leave this empty (0) or input single random tokens. |
| 4 | `nums/length fullname` | Continuous [0, 1] | Ratio of numeric digits to total characters in full name. | Legitimate human names rarely contain numeric digits. |
| 5 | `name==username` | Binary (0 / 1) | Equals 1 if the displayed full name matches username verbatim; else 0. | Automated registration scripts often duplicate username into the name field. |
| 6 | `description length` | Integer (≥ 0) | Character count of the account bio/description. | Spammers and throwaway bots often have blank or minimal descriptions. |
| 7 | `external URL` | Binary (0 / 1) | Whether a link/URL is provided in the bio (1 = Yes, 0 = No). | Presence or absence of website links in profiles. |
| 8 | `private` | Binary (0 / 1) | Whether the profile is private (1 = Private, 0 = Public). | Many commercial fake/scraper profiles remain public to maximize impressions. |
| 9 | `#posts` | Integer (≥ 0) | Number of media posts uploaded by the account. | Fake accounts typically have very low post counts or sporadic bursts. |
| 10 | `#followers` | Integer (≥ 0) | Total count of followers subscribing to this profile. | Key signal for account authority and engagement. |
| 11 | `#follows` | Integer (≥ 0) | Total count of accounts this profile is following. | Aggressive follow-for-follow bot behaviors show disproportionately high follows. |
| 12 | `followers_following_ratio`| Continuous (≥ 0) | Ratio calculated as `#followers / #follows`. | Accounts following thousands with single-digit followers strongly correlate with bots. |

---

## 3. Feature Engineering & Preprocessing

Located in `prepare_data(df)`:

1. **Infinity & Anomaly Cleaning**:
   ```python
   df = df.copy().replace([np.inf, -np.inf], np.nan)
   ```
2. **Target Sanitization**:
   - Converts `fake` target column to numeric with coercion.
   - Drops rows missing targets and filters strictly to binary labels $\{0, 1\}$.
3. **Missing Value Imputation**:
   - Imputes missing values with column medians, followed by fallback to `0`.
4. **Standardization & Outlier Clamping**:
   - Scaled using `sklearn.preprocessing.StandardScaler`:
     $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
   - Values are clamped to $[-10, 10]$ (`np.clip(..., -10, 10)`) to guard against extreme outliers from celebrity accounts with millions of followers.

---

## 4. Graph Construction: Feature-Similarity Topology

Unlike citation or physical social graphs where edge lists exist, tabular datasets lack explicit edges. The system constructs a **$k$-Nearest-Neighbor Feature-Similarity Graph** via `build_graph(X_scaled, k)`:

1. **Connectivity Graph**: Uses `kneighbors_graph` from `sklearn.neighbors` with $k$ neighbors (default $k=6$).
2. **Graph Symmetrization**: Makes the graph undirected:
   $$A_{sym} = \max(A, A^T)$$
3. **Self-Loop Addition**: Sets $A_{ii} = 1.0$ so nodes retain their own attributes during message passing.
4. **Dual Representations Exported**:
   - **Normalized Sparse Laplacian $\hat{A}$** for standard GCN convolutions:
     $$\hat{A} = D^{-1/2} A D^{-1/2}$$
   - **Edge Index Tensor (`edge_index`)** of shape $[2, |E|]$ for Graph Attention Networks (GAT) and GNNExplainer mask optimization.

---

## 5. Deep Learning Architectures: GAT vs GCN

Users can switch between two graph neural network architectures in the web interface:

### Architecture A: Graph Attention Network (GAT — SpaCEy Inspired)
Inspired by the **Saez Lab's SpaCEy framework** (*Nature Communications*):
Instead of fixed degree weighting, GAT dynamically learns how much attention node $i$ should pay to neighbor $j$:
$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j^{(l)}\right)$$
- **Benefits**: Focuses attention on true signals of automation and discounts irrelevant neighbors.
- **Layer 1**: 12 inputs $\to$ 32 hidden units with ELU activation and dropout ($p=0.2$).
- **Layer 2**: 32 hidden units $\to$ 2 output classes (Real, Fake).

### Architecture B: Graph Convolutional Network (GCN Baseline)
Standard Kipf & Welling GCN:
$$\mathbf{H}^{(1)} = \text{Dropout}\left(\text{ReLU}\left(\hat{A} \mathbf{X} \mathbf{W}^{(1)}\right), p=0.35\right)$$
$$\mathbf{Z} = \hat{A} \mathbf{H}^{(1)} \mathbf{W}^{(2)}$$

---

## 6. Training & Adaptive Optimization Pipeline

Located in `train_gnn(...)`:

- **Stratified Partitioning**: 80/20 train/test split preserving exact class balance.
- **Adaptive Learning Rate Scheduling (SpaCEy Inspired)**:
  - Uses `torch.optim.lr_scheduler.ReduceLROnPlateau` with decay factor $0.5$ and patience $5$.
  - Monitors training loss and reduces learning rate when plateaus occur, preventing overshooting.
- **Optimizer**: Adam ($\text{lr} = 0.01$, $L_2$ weight decay $= 5 \times 10^{-4}$).
- **Validation Metrics**:
  - Accuracy: **~95.2% – 95.6%**
  - Precision: **~96.3%**
  - Recall: **~94.8%**
  - F1-Score: **~95.5%**

---

## 7. Inference Engine for New User Inputs

When an ad-hoc account is entered in the web UI:
1. Normalizes the feature vector using the pre-fitted `StandardScaler`.
2. Locates the $k$-nearest profiles in latent feature space via Euclidean distance.
3. Computes distance-weighted soft aggregation ($w_i = \frac{1}{d_i + 10^{-6}}$).
4. Aggregates neighbor model probabilities to produce `fake_probability`.

---

## 8. Dual Explainability Suite (GNNExplainer & Saliency)

Users can select their preferred explainability engine:

### Engine 1: GNNExplainer (SpaCEy Mask Optimization)
Directly adapted from `bin/explainer.py` in SpaCEy:
- Sets up a local star subgraph connecting the user profile to its $k$ nearest neighbors.
- Jointly optimizes:
  - **Feature Mask** $\mathbf{m}_{\text{feat}} \in [0, 1]^{12}$
  - **Edge Mask** $\mathbf{m}_{\text{edge}} \in [0, 1]^k$
- Minimizes cross-entropy with the predicted class subject to **$L_1$ sparsity** and **entropy penalties** $\mathcal{H}(m)$:
  $$\mathcal{L}_{\text{expl}} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda_{\text{size}} (\|\mathbf{m}_{\text{feat}}\|_1 + \|\mathbf{m}_{\text{edge}}\|_1) + \lambda_{\text{ent}} (\mathcal{H}(\mathbf{m}_{\text{feat}}) + \mathcal{H}(\mathbf{m}_{\text{edge}}))$$
- The entropy term pushes mask values decisively toward 0 or 1, producing sharp, high-confidence explanations.

### Engine 2: Gradient × Input (Fast Saliency)
- Computes first-order gradient of predicted class logit with respect to input features: $|\nabla_{\mathbf{x}} Z_c \cdot \mathbf{x}|$.

---

## 9. Topological Motif & Bot Syndicate Discovery

Adapted from SpaCEy's functional spatial motif discovery:
- In `analyze_subgraph_motif`, the system inspects the target account's neighborhood composition:
  - **🤖 Bot Syndicate / Farm Motif** (Weighted Bot Affinity $\ge 65\%$): The account clusters tightly with synthetic bot profiles. High probability of being part of an automated follower farm.
  - **👤 Authentic Human Community Motif** (Weighted Bot Affinity $\le 35\%$): The account clusters within organic human user distributions.
  - **⚖️ Borderline / Mixed Neighborhood** ($35\% < \text{Affinity} < 65\%$): Sits on the boundary between authentic and bot distributions.

---

## 10. Web Application Structure & User Interface

The frontend is built using **Streamlit** in `app.py`:
- **Sidebar**:
  - GNN Architecture Selector (GAT vs GCN).
  - Explainability Engine Selector (GNNExplainer vs Saliency).
  - Hyperparameter controls ($k$, test size, epochs, hidden units, seed).
- **Tab 1: 🔍 Check a Profile**:
  - **⚡ Automated Link Input**: Paste any profile URL (e.g., `https://www.instagram.com/cristiano/`) or `@username`.
  - Automatic extraction of follower count, following count, post count, bio length, digit ratios, and name matching.
  - Live account summary card showing avatar, bio, and quick stats.
  - 12-field input form pre-populated with extracted values, allowing manual adjustments if desired.
  - Prediction badge, Fake Score (%), and Model Accuracy.
  - Subgraph Motif Banner (identifying Bot Syndicates).
  - Feature attribution horizontal bar chart.
  - Neighbor influence distribution chart.
  - Detailed topological neighbors table.
- **Tab 2: 📈 Model Performance**:
  - Metric cards (Accuracy, Precision, Recall, F1).
  - Loss progression curve showing `ReduceLROnPlateau` decay steps.
  - Confusion matrix heatmap.
  - Scikit-Learn classification report.
- **Tab 3: 📂 About Dataset**:
  - Dataset overview stats and 15-row preview.
- **Tab 4: 🧬 SpaCEy Methods & Research**:
  - Comprehensive guide explaining the Saez Lab GAT, GNNExplainer, and Motif methods.

---

## 11. Automated Profile Link Extraction Engine (`url_extractor.py`)

The application includes an automated feature extractor that takes an Instagram profile link or handle and parses all 12 input features without requiring manual user data entry:

### How it Works:
1. **URL Sanitization (`clean_instagram_input`)**:
   Strips tracking parameters (`?igsh=...`), protocol prefixes, and trailing slashes to isolate the canonical Instagram handle.
2. **Metadata Crawler Request**:
   Queries Instagram's public SSR OpenGraph endpoint using crawler headers (`facebookexternalhit` / `Googlebot`).
3. **Regex Attribute Parsing**:
   - Parses `og:description` and `meta[name=description]` for Followers (`687M`), Following (`307`), and Posts (`8,617`).
   - Extracts displayed full name from `og:title`.
   - Extracts account biography from the unescaped meta description.
   - Detects profile picture existence via `og:image`.
   - Scans bio for external URLs (linktr.ee, bit.ly, http links).
   - Detects account privacy status.
4. **Calculated Ratios**:
   - Automatically computes `nums/length username` and `nums/length fullname`.
   - Computes `fullname words` and checks `name == username`.
   - Calculates `followers_following_ratio = #followers / max(#follows, 1)`.
5. **Instant Pre-population**:
   Auto-fills all 12 numeric fields in the web form and renders a profile card before running GNN inference.


---

## 12. How to Run & Configure

### Prerequisites
- Python 3.10+
- PyTorch (`torch`)
- Streamlit (`streamlit`)
- Scikit-Learn (`scikit-learn`)
- SciPy (`scipy`)
- Pandas (`pandas`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

### Running the Application

```bash
streamlit run app.py
```

The web application opens at:
```
http://localhost:8501
```
