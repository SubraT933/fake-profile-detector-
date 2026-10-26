"""Production Web Application for Fake Profile & Bot Detection."""

import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is in the module search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_profile_detector.config import DATASETS_DIR, REPORTS_DIR, ModelConfig
from fake_profile_detector.data import load_dataset, prepare_dataset
from fake_profile_detector.extractor import extract_profile_features
from fake_profile_detector.pipeline import (
    analyze_community_motif,
    explain_profile,
    predict_profile,
    train_model,
)

st.set_page_config(
    page_title="Social Account Verification & Bot Detection Platform",
    page_icon="none",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional executive design styling (no emojis, clean lines, high contrast)
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    .app-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin: 0.25rem 0 0 0;
    }
    .profile-card {
        padding: 1.25rem;
        border-radius: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    .verdict-box {
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .verdict-danger {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }
    .verdict-success {
        background-color: #f0fdf4;
        border-left: 4px solid #10b981;
        color: #166534;
    }
    .verdict-warning {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-synthetic { background: #fee2e2; color: #b91c1c; }
    .badge-genuine { background: #dcfce7; color: #15803d; }
    .stMetric label { font-size: 0.85rem !important; color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1 class="app-title">Social Account Verification & Bot Detection Platform</h1>
    <p class="app-subtitle">Graph Neural Network inference and topological neighborhood analysis for social media authentication.</p>
</div>
""", unsafe_allow_html=True)


# Persistent session states
if "trained_artifact" not in st.session_state:
    st.session_state.trained_artifact = None
if "artifact_config_key" not in st.session_state:
    st.session_state.artifact_config_key = None
if "current_profile_data" not in st.session_state:
    st.session_state.current_profile_data = None
if "inference_result" not in st.session_state:
    st.session_state.inference_result = None
if "motif_result" not in st.session_state:
    st.session_state.motif_result = None
if "attribution_df" not in st.session_state:
    st.session_state.attribution_df = None


# Sidebar controls
with st.sidebar:
    st.subheader("Model Configuration")
    architecture = st.selectbox(
        "GNN Architecture",
        options=["Graph Attention Network (GAT)", "Graph Convolutional Network (GCN)"],
        index=0,
        help="GAT uses parameterized dynamic edge attention; GCN uses static degree normalization.",
    )
    selected_arch = "GAT" if "GAT" in architecture else "GCN"

    explainer_type = st.selectbox(
        "Explainability Method",
        options=["GNNExplainer (Mask Optimization)", "Gradient-Times-Input Saliency"],
        index=0,
        help="GNNExplainer optimizes subgraph masks with mutual information; Saliency computes 1st order gradients.",
    )
    selected_explainer = "gnn_explainer" if "GNNExplainer" in explainer_type else "saliency"

    st.markdown("---")
    st.subheader("Reference Dataset")
    available_datasets = sorted([f.name for f in DATASETS_DIR.glob("*.csv") if not f.name.startswith("user_provided") and not f.name.startswith("dataset_1")])
    default_idx = available_datasets.index("Instagram_5k.csv") if "Instagram_5k.csv" in available_datasets else 0
    selected_dataset = st.selectbox(
        "Training Corpus",
        options=available_datasets,
        index=default_idx,
    )

    st.markdown("---")
    st.subheader("Hyperparameters")
    k_neighbors = st.slider("Graph Neighbors (k)", min_value=3, max_value=16, value=8)
    epochs = st.slider("Epochs", min_value=10, max_value=100, value=40, step=10)
    hidden_dim = st.select_slider("Hidden Units", options=[16, 32, 64], value=32)
    random_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)


@st.cache_data(show_spinner=False)
def get_cached_raw_data(dataset_filename: str) -> pd.DataFrame:
    return load_dataset(dataset_filename)


try:
    df_raw = get_cached_raw_data(selected_dataset)
except Exception as exc:
    st.error(f"Failed to load dataset '{selected_dataset}': {exc}")
    st.stop()


def ensure_model_trained():
    """Ensure the model artifact is current with sidebar configuration."""
    current_key = (selected_dataset, selected_arch, k_neighbors, epochs, hidden_dim, int(random_seed))
    if st.session_state.trained_artifact is None or st.session_state.artifact_config_key != current_key:
        with st.spinner("Training Graph Neural Network on reference dataset..."):
            cfg = ModelConfig(
                model_type=selected_arch,
                k_neighbors=k_neighbors,
                epochs=epochs,
                hidden_dim=hidden_dim,
                random_seed=int(random_seed),
            )
            artifact = train_model(df_raw, config=cfg)
            st.session_state.trained_artifact = artifact
            st.session_state.artifact_config_key = current_key
    return st.session_state.trained_artifact


def run_account_analysis(features: Dict[str, float]):
    """Execute inference, community motif detection, and feature attribution."""
    artifact = ensure_model_trained()
    pred = predict_profile(artifact, features, k=k_neighbors)
    motif = analyze_community_motif(artifact, pred)
    expl_df, _ = explain_profile(artifact, pred, method=selected_explainer)

    st.session_state.inference_result = pred
    st.session_state.motif_result = motif
    st.session_state.attribution_df = expl_df


# Main navigation tabs
tab_verify, tab_benchmark, tab_dataset, tab_methods = st.tabs([
    "Account Verification",
    "Model Benchmarks",
    "Dataset Profiles",
    "Technical Architecture",
])


# ==============================================================================
# TAB 1: Account Verification & Inference
# ==============================================================================
with tab_verify:
    st.markdown("### Profile Ingestion & Verification")
    st.markdown("Enter an Instagram profile link or handle to analyze its behavioral features and detect synthetic indicators.")

    c_input, c_btn = st.columns([4, 1.2])
    with c_input:
        profile_input = st.text_input(
            "Account URL or Username",
            placeholder="e.g. https://www.instagram.com/cristiano/ or @nasa",
            label_visibility="collapsed",
            key="url_input",
        )
    with c_btn:
        execute_clicked = st.button("Run Verification", type="primary", use_container_width=True)

    # Reference presets for testing
    p_label, p1, p2, p3 = st.columns([1.5, 1.2, 1.2, 1.5])
    with p_label:
        st.caption("Verification test samples:")
    with p1:
        if st.button("Cristiano", use_container_width=True):
            profile_input = "cristiano"
            execute_clicked = True
    with p2:
        if st.button("NASA", use_container_width=True):
            profile_input = "nasa"
            execute_clicked = True
    with p3:
        if st.button("Automated Bot", use_container_width=True):
            profile_input = "sample_spam_bot_9812"
            execute_clicked = True

    if execute_clicked and profile_input:
        features, meta, err = extract_profile_features(profile_input)
        if err:
            st.error(err)
        else:
            st.session_state.current_profile_data = (features, meta)
            run_account_analysis(features)

    # Render Profile Metadata Card
    if st.session_state.current_profile_data is not None:
        features, meta = st.session_state.current_profile_data
        privacy_text = "Private Account" if meta.get("is_private") else "Public Account"
        status_badge = '<span class="badge-pill badge-synthetic">Synthetic Alert</span>' if st.session_state.inference_result and st.session_state.inference_result["is_fake"] else '<span class="badge-pill badge-genuine">Authentic</span>'

        st.markdown(f"""
        <div class="profile-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h3 style="margin: 0; color: #0f172a; font-size: 1.25rem;">{meta.get('fullname') or meta['username']}</h3>
                    <p style="margin: 0.15rem 0 0 0; color: #64748b; font-size: 0.9rem;">@{meta['username']} &bull; {privacy_text}</p>
                </div>
                <div>{status_badge}</div>
            </div>
            <p style="margin: 0.5rem 0 0.8rem 0; color: #334155; font-size: 0.9rem; font-style: italic;">
                "{meta.get('bio') or 'No profile biography provided.'}"
            </p>
            <div style="display: flex; gap: 2.5rem; font-size: 0.9rem; color: #1e293b; border-top: 1px solid #e2e8f0; padding-top: 0.6rem;">
                <div><b>Followers:</b> {int(features.get('#followers', 0)):,}</div>
                <div><b>Following:</b> {int(features.get('#follows', 0)):,}</div>
                <div><b>Posts:</b> {int(features.get('#posts', 0)):,}</div>
                <div><b>Ratio (F/F):</b> {features.get('followers_following_ratio', 0.0):.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Collapsible manual adjustments
        with st.expander("Feature Parameter Breakdown & Adjustments", expanded=False):
            feature_cols = [c for c in features.keys()]
            adjusted_vals = {}
            col_chunks = st.columns(3)
            for idx, col in enumerate(feature_cols):
                val = float(features[col])
                with col_chunks[idx % 3]:
                    adjusted_vals[col] = st.number_input(
                        col,
                        value=float(val),
                        key=f"param_{col}",
                    )
            if st.button("Re-evaluate with Modified Features", type="secondary"):
                st.session_state.current_profile_data = (adjusted_vals, meta)
                run_account_analysis(adjusted_vals)
                st.rerun()

    # Render Verification Results
    if st.session_state.inference_result is not None:
        pred = st.session_state.inference_result
        motif = st.session_state.motif_result
        artifact = st.session_state.trained_artifact

        st.markdown("---")
        st.markdown("### Classification Verdict & Topological Assessment")

        m1, m2, m3, m4 = st.columns(4)
        verdict_str = "Synthetic / Automated" if pred["is_fake"] else "Authentic / Human"
        m1.metric("Verification Verdict", verdict_str)
        m2.metric("Synthetic Risk Probability", f"{pred['fake_probability']:.1%}")
        m3.metric("Model Test Accuracy", f"{artifact['metrics']['accuracy']:.1%}")
        m4.metric("Active Architecture", artifact["model_type"])

        st.progress(float(np.clip(pred["fake_probability"], 0.0, 1.0)))

        # Community Motif Box
        variant_class = f"verdict-{motif['badge_variant']}"
        st.markdown(f"""
        <div class="verdict-box {variant_class}">
            <strong>Topological Classification: {motif['cluster_category']}</strong><br/>
            {motif['description']}<br/>
            <span style="font-size: 0.85rem; color: #475569; margin-top: 0.35rem; display: inline-block;">
                Local Neighborhood: {motif['fake_count']} synthetic / {motif['genuine_count']} genuine profiles &bull; Weighted Synthetic Affinity: {motif['weighted_affinity']:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Feature Attribution")
            st.caption("Attribution scores derived from local mask optimization:")
            att_df = st.session_state.attribution_df
            top_att = att_df.tail(8) if len(att_df) > 8 else att_df
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.barh(top_att["feature"], top_att["importance"], color="#2563eb", height=0.6)
            ax.set_xlabel("Relative Attribution Score")
            ax.grid(axis="x", alpha=0.2, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.markdown("##### Neighbor Influence Distribution")
            st.caption("Proximity influence weights of closest training nodes:")
            weights = pred["weights"]
            fig, ax = plt.subplots(figsize=(6, 3.8))
            neighbor_indices = pred["nearest_indices"]
            colors = ["#ef4444" if artifact["y"][idx] == 1 else "#10b981" for idx in neighbor_indices]
            ax.bar([f"Node {i+1}" for i in range(len(weights))], weights, color=colors, width=0.6)
            ax.set_ylabel("Normalized Influence")
            ax.grid(axis="y", alpha=0.2, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("##### Closest Dataset Neighborhood Profiles")
        st.dataframe(motif["neighbor_table"], use_container_width=True)
    else:
        st.info("Enter an Instagram handle or profile URL above and click 'Run Verification' to initiate inference.")


# ==============================================================================
# TAB 2: Multi-Model Benchmark Leaderboard
# ==============================================================================
with tab_benchmark:
    st.markdown("### Classifier Performance Leaderboard")
    st.markdown("Comparative evaluation of standard machine learning and graph deep learning algorithms on the active dataset:")

    benchmark_csv = REPORTS_DIR / f"benchmark_{Path(selected_dataset).stem}.csv"
    if benchmark_csv.is_file():
        bm_df = pd.read_csv(benchmark_csv)
        display_cols = ["rank", "model", "family", "accuracy", "f1_score", "precision", "recall", "roc_auc", "train_time_seconds"]
        cols = [c for c in display_cols if c in bm_df.columns]
        st.dataframe(
            bm_df[cols].style.format({
                "accuracy": "{:.1%}",
                "f1_score": "{:.3f}",
                "precision": "{:.1%}",
                "recall": "{:.1%}",
                "roc_auc": "{:.4f}",
                "train_time_seconds": "{:.3f}s",
            }),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(10, 4.2))
        plot_df = bm_df.head(10).iloc[::-1]
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos - 0.2, plot_df["accuracy"], height=0.35, label="Accuracy", color="#2563eb")
        ax.barh(y_pos + 0.2, plot_df["f1_score"], height=0.35, label="F1-Score", color="#10b981")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["model"])
        ax.set_xlabel("Score")
        ax.set_xlim(0.65, 1.02)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"No saved benchmark found for {selected_dataset}. Run `python run_benchmark.py --dataset {selected_dataset}` to generate.")

    st.markdown("---")
    st.markdown("### Active GNN Training Convergence")
    artifact = st.session_state.trained_artifact
    if artifact is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Loss Trajectory ({artifact['model_type']})**")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(artifact["losses"], color="#2563eb", linewidth=1.8, label="Cross-Entropy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.markdown("**Holdout Partition Metrics**")
            metrics = artifact["metrics"]
            st.write(pd.DataFrame([{
                "Accuracy": f"{metrics['accuracy']:.3f}",
                "Balanced Accuracy": f"{metrics['balanced_accuracy']:.3f}",
                "Precision": f"{metrics['precision']:.3f}",
                "Recall": f"{metrics['recall']:.3f}",
                "F1-Score": f"{metrics['f1_score']:.3f}",
                "ROC-AUC": f"{metrics['roc_auc']:.3f}",
            }]))


# ==============================================================================
# TAB 3: Dataset Profiles
# ==============================================================================
with tab_dataset:
    st.markdown(f"### Dataset Profile: {selected_dataset}")
    clean_df, _, y_all, feature_cols = prepare_dataset(df_raw)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(clean_df):,}")
    c2.metric("Feature Dimensions", len(feature_cols))
    fake_pct = np.mean(y_all) * 100
    c3.metric("Synthetic Distribution", f"{int(np.sum(y_all)):,} ({fake_pct:.1f}%)")

    st.markdown("##### Sample Records")
    st.dataframe(clean_df.head(20), use_container_width=True)


# ==============================================================================
# TAB 4: Technical Architecture
# ==============================================================================
with tab_methods:
    st.markdown("""
    ### System Architecture & Mathematical Foundations

    The platform implements Graph Neural Networks (GNN) and topological graph analysis to identify fraudulent social media entities.

    #### 1. Topology Construction
    Continuous profile feature vectors are mapped into a $k$-nearest-neighbor graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
    - Adjacency matrix $A$ is symmetrized: $A = \max(A_{raw}, A_{raw}^T) + I_N$.
    - Normalized Laplacian propagation for GCN: $\tilde{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.

    #### 2. Graph Attention Mechanism (GAT)
    Instead of fixed uniform weights, the Graph Attention Network learns parametric attention coefficients $\alpha_{ij}$:
    $$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i \,\|\, \mathbf{W} h_j]\right)$$
    $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

    #### 3. Out-of-Graph Inductive Inference
    When an external query profile $x_q$ is evaluated:
    - The profile is projected into the learned latent metric space via the fitted feature scaler.
    - An inverse-distance weighted ensemble calculates the synthetic risk probability across its $k$ nearest manifold neighbors:
    $$P(\text{fake} \mid x_q) = \sum_{j \in \mathcal{N}_k(x_q)} w_j \cdot P(\text{fake} \mid x_j), \quad w_j = \frac{\|x_q - x_j\|^{-1}}{\sum_m \|x_q - x_m\|^{-1}}$$

    #### 4. Subgraph Mask Explainability
    Local feature and edge attributions are optimized through continuous relaxation masks $m_{feat} \in [0, 1]^d$ and $m_{edge} \in [0, 1]^k$:
    $$\min_{m_{feat}, m_{edge}} \mathcal{L}_{CE}(y, \hat{y}) + \lambda_1 \left(\|m_{feat}\|_1 + \|m_{edge}\|_1\right) + \lambda_2 \left(\mathcal{H}(m_{feat}) + \mathcal{H}(m_{edge})\right)$$
    where $\mathcal{H}(m) = -m \log m - (1-m)\log(1-m)$ is the entropy penalty enforcing binary mask convergence.
    """)

st.markdown("---")
st.caption("Enterprise Social Fraud Detection Platform • Core Engine v1.0.0")
