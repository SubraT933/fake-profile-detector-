import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import importlib
import ml_model
importlib.reload(ml_model)
from ml_model import (
    prepare_data, train_gnn, predict_user_profile,
    explain_user_profile, analyze_subgraph_motif
)

import url_extractor
importlib.reload(url_extractor)
from url_extractor import extract_features_from_instagram_url, clean_instagram_input, SAMPLE_PROFILES

st.set_page_config(
    page_title="Fake Profile Detector — SpaCEy Enhanced",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #312e81 100%);
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
}
.hero h1 {color: white; margin-bottom: .4rem; font-size: 2.1rem; font-weight: 700;}
.hero p {color: #cbd5e1; margin-bottom: 0; font-size: 1.05rem;}
.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
    background: rgba(255, 255, 255, 0.15);
    color: #e2e8f0;
}
.profile-card {
    padding: 1.3rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    border: 1px solid #bfdbfe;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🛡️ Fake Instagram Profile Detector</h1>
<p>Paste any profile link or username to <b>instantly extract features and detect fake accounts</b> using Graph Attention Networks (GAT).</p>
<span class="badge">⚡ Instant Link Auto-Extraction</span>
<span class="badge">🧬 SpaCEy Graph Attention (GAT)</span>
<span class="badge">🔍 Mask-Based Explainability</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Model Architecture & Settings")
    model_choice = st.selectbox(
        "GNN Architecture",
        ["GAT (Graph Attention — SpaCEy Inspired)", "GCN (Kipf & Welling Baseline)"],
        index=0,
        help="GAT dynamically weighs edges via attention; GCN uses fixed degree normalization."
    )
    model_type = "GAT" if "GAT" in model_choice else "GCN"

    explainer_choice = st.selectbox(
        "Explainability Method",
        ["GNNExplainer (SpaCEy Mask Optimization)", "Gradient × Input (Fast Saliency)"],
        index=0,
        help="GNNExplainer optimizes mutual information & entropy over masks; Saliency uses 1st order gradients."
    )
    explainer_method = "gnn_explainer" if "GNNExplainer" in explainer_choice else "saliency"

    st.divider()
    st.subheader("Hyperparameters")
    k = st.slider("Graph Neighbors (k)", 3, 15, 8, help="Number of closest profiles in feature space (optimal k=8 based on sensitivity analysis).")
    test_size = st.slider("Test Partition Fraction", 0.15, 0.35, 0.20, 0.05)
    epochs = st.slider("Training Epochs", 20, 120, 50, 10)
    hidden_dim = st.select_slider("Hidden Units", options=[16, 32, 64], value=32)
    seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)

@st.cache_data
def load_csv():
    return pd.read_csv("Instagram.csv")

try:
    df_raw = load_csv()
    df, feature_cols = prepare_data(df_raw)
except Exception as e:
    st.error(f"Could not load Instagram.csv: {e}")
    st.info("Keep Instagram.csv in the same folder as app.py and ml_model.py.")
    st.stop()

if "model_result" not in st.session_state:
    st.session_state.model_result = None
if "trained_settings" not in st.session_state:
    st.session_state.trained_settings = None
if "fetched_features" not in st.session_state:
    st.session_state.fetched_features = None
if "fetched_meta" not in st.session_state:
    st.session_state.fetched_meta = None
if "user_prediction" not in st.session_state:
    st.session_state.user_prediction = None


def execute_analysis(profile_dict):
    """Train/fetch model and run prediction, motif analysis, and explainability."""
    current_settings = (k, test_size, epochs, hidden_dim, int(seed), model_type)
    needs_retrain = (
        st.session_state.model_result is None or
        st.session_state.trained_settings != current_settings
    )

    if needs_retrain:
        result = train_gnn(
            df, k=k, test_size=test_size, epochs=epochs,
            hidden_dim=hidden_dim, seed=int(seed), model_type=model_type
        )
        st.session_state.model_result = result
        st.session_state.trained_settings = current_settings
    else:
        result = st.session_state.model_result

    prediction = predict_user_profile(result, profile_dict, k=k)
    motif_data = analyze_subgraph_motif(result, profile_dict, k=k)
    explanation_df, edge_weights = explain_user_profile(
        result, profile_dict, k=k, method=explainer_method
    )

    st.session_state.user_prediction = prediction
    st.session_state.user_profile = profile_dict
    st.session_state.user_motif = motif_data
    st.session_state.user_explanation = explanation_df
    st.session_state.user_edge_weights = edge_weights


tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Check a Profile",
    "📈 Model Performance",
    "📂 About Dataset",
    "🧬 SpaCEy Methods & Research"
])

# ==============================================================================
# TAB 1: Check a Profile
# ==============================================================================
with tab1:
    st.subheader("⚡ Automated Instagram Profile Analysis")
    st.write("Paste any public Instagram link or handle below to automatically extract data and run deep GNN analysis.")

    col_input, col_fetch = st.columns([4, 1.3])
    with col_input:
        profile_url_input = st.text_input(
            "Paste Instagram Profile URL or @Username",
            placeholder="e.g. https://www.instagram.com/cristiano/ or @nasa",
            label_visibility="collapsed",
            key="url_input_box"
        )
    with col_fetch:
        fetch_triggered = st.button("🚀 Analyze Profile Link", type="primary", use_container_width=True)

    # Quick testing presets
    c_label, c_btn1, c_btn2, c_btn3 = st.columns([1.3, 1.2, 1.2, 1.8])
    with c_label:
        st.caption("Quick one-click samples:")
    with c_btn1:
        if st.button("👤 Cristiano", use_container_width=True):
            profile_url_input = "cristiano"
            fetch_triggered = True
    with c_btn2:
        if st.button("🚀 NASA", use_container_width=True):
            profile_url_input = "nasa"
            fetch_triggered = True
    with c_btn3:
        if st.button("🤖 Sample Spam Bot", use_container_width=True):
            profile_url_input = "sample_spam_bot_9812"
            fetch_triggered = True

    # When URL or quick sample is submitted, extract AND immediately execute analysis!
    if fetch_triggered and profile_url_input:
        with st.spinner("Fetching profile metadata and running Graph Attention Network..."):
            features_dict, meta_dict, err = extract_features_from_instagram_url(profile_url_input)
            if err:
                st.error(err)
            else:
                st.session_state.fetched_features = features_dict
                st.session_state.fetched_meta = meta_dict
                try:
                    execute_analysis(features_dict)
                    st.toast(f"Analyzed @{meta_dict['username']} successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    # Display Fetched Profile Summary Card
    if st.session_state.fetched_meta is not None:
        meta = st.session_state.fetched_meta
        f_feat = st.session_state.fetched_features
        st.markdown(f"""
        <div class="profile-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 2.2rem;">{'👤' if not meta.get('avatar_url') else '🖼️'}</div>
                    <div>
                        <h3 style="margin: 0; color: #0f172a;">{meta.get('fullname') or meta['username']}</h3>
                        <p style="margin: 0; color: #64748b; font-size: 0.95rem;">@{meta['username']} &bull; {'🔒 Private' if meta.get('is_private') else '🌐 Public'}</p>
                    </div>
                </div>
                <span style="background: #22c55e; color: white; padding: 0.35rem 0.85rem; border-radius: 9999px; font-weight: 600; font-size: 0.85rem;">
                    ⚡ Auto-Extracted & Analyzed
                </span>
            </div>
            <p style="margin: 0.4rem 0 0.8rem 0; color: #334155; font-size: 0.95rem; font-style: italic;">
                "{meta.get('bio') or 'No biography provided.'}"
            </p>
            <div style="display: flex; gap: 2rem; font-size: 0.95rem; color: #1e293b; border-top: 1px solid #dbeafe; padding-top: 0.6rem;">
                <div><b>Followers:</b> {int(f_feat['#followers']):,}</div>
                <div><b>Following:</b> {int(f_feat['#follows']):,}</div>
                <div><b>Posts:</b> {int(f_feat['#posts']):,}</div>
                <div><b>Follower/Following Ratio:</b> {f_feat['followers_following_ratio']:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Optional collapsible for fine-tuning individual features
    with st.expander("🛠️ View / Fine-Tune Extracted Features (Optional)", expanded=False):
        st.write("You can inspect or adjust the 12 numerical features used by the GNN:")
        feature_cols = [c for c in feature_cols if c != "fake"]
        defaults = df[feature_cols].median(numeric_only=True).to_dict()
        profile_form_vals = {}
        descriptions = {
            "profile pic": "Has profile picture (0 = no, 1 = yes)",
            "nums/length username": "Numbers-to-username-length ratio",
            "fullname words": "Number of words in full name",
            "nums/length fullname": "Numbers-to-fullname-length ratio",
            "name==username": "Name matches username (0/1)",
            "description length": "Bio/description length",
            "external URL": "Has external URL (0/1)",
            "private": "Private account (0/1)",
            "#posts": "Number of posts",
            "#followers": "Number of followers",
            "#follows": "Number of accounts followed",
            "followers_following_ratio": "Followers divided by following"
        }

        with st.form("manual_adjust_form"):
            cols = st.columns(3)
            for i, col in enumerate(feature_cols):
                if st.session_state.fetched_features and col in st.session_state.fetched_features:
                    val = float(st.session_state.fetched_features[col])
                else:
                    val = float(defaults.get(col, 0))

                low = float(df[col].min())
                high = float(df[col].max())
                if low == high:
                    low, high = val - 1000000, val + 1000000
                else:
                    low = min(low, val)
                    high = max(high, val)

                with cols[i % 3]:
                    profile_form_vals[col] = st.number_input(
                        descriptions.get(col, col),
                        min_value=float(low),
                        max_value=float(high),
                        value=float(val),
                        key=f"field_{col}",
                        help=f"Dataset column: {col}"
                    )
            reanalyze_submitted = st.form_submit_button(
                "🔄 Re-Analyze with Adjusted Values",
                type="secondary",
                use_container_width=True
            )

        if reanalyze_submitted:
            with st.spinner("Re-running GNN analysis with adjusted features..."):
                try:
                    execute_analysis(profile_form_vals)
                    st.session_state.fetched_features = profile_form_vals.copy()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during re-analysis: {e}")

    # Display Analysis Results
    if st.session_state.get("user_prediction") is not None:
        prediction = st.session_state.user_prediction
        motif_data = st.session_state.user_motif
        explanation_df = st.session_state.user_explanation
        res = st.session_state.model_result

        st.divider()
        st.subheader("🎯 Analysis Results & Detection Decision")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Verdict",
            "FAKE-LIKE" if prediction["predicted_class"] == 1 else "REAL-LIKE",
            delta="High Bot Risk" if prediction["predicted_class"] == 1 else "Authentic / Human",
            delta_color="inverse"
        )
        c2.metric("Estimated Fake Risk", f"{prediction['fake_probability']:.1%}")
        c3.metric("Model Test Accuracy", f"{res['accuracy']:.1%}")
        c4.metric("Architecture", res.get("model_type", "GAT"))

        st.progress(float(np.clip(prediction["fake_probability"], 0, 1)))

        # Subgraph Motif Discovery (SpaCEy Inspired)
        st.markdown(f"""
        <div style="padding: 1.1rem; border-radius: 12px; border-left: 6px solid {motif_data['motif_color']}; background: #f8fafc; margin: 1.2rem 0;">
            <h4 style="margin: 0 0 0.3rem 0; color: #0f172a;">{motif_data['motif_type']}</h4>
            <p style="margin: 0; color: #475569; font-size: 0.95rem;">{motif_data['motif_desc']}</p>
            <div style="margin-top: 0.6rem; font-size: 0.9rem; color: #334155;">
                <b>Topological Composition:</b> {motif_data['fake_neighbor_count']} Fake / {motif_data['real_neighbor_count']} Real neighbors 
                &bull; <b>Weighted Bot Affinity:</b> {motif_data['weighted_fake_affinity']:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader(f"Feature Attribution ({'GNNExplainer' if explainer_method == 'gnn_explainer' else 'Gradient × Input'})")
            st.caption("Which attributes tipped the model toward this verdict:")
            imp = explanation_df.sort_values("Importance", ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.barh(imp["Feature"], imp["Importance"], color="#3b82f6")
            ax.set_xlabel("Relative Importance Score")
            ax.grid(axis="x", alpha=0.3, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.subheader("Neighbor Influence Distribution")
            st.caption("SpaCEy-inspired edge weight attribution across closest network neighbors:")
            weights = motif_data["weights"]
            fig, ax = plt.subplots(figsize=(6, 4.2))
            bars = ax.bar(
                [f"N-{i+1}" for i in range(len(weights))],
                weights,
                color=["#ef4444" if res["y"][idx] == 1 else "#10b981" for idx in motif_data["nearest"]]
            )
            ax.set_ylabel("Edge Influence Weight")
            ax.set_xlabel("Dataset Neighbor (Red = Fake, Green = Real)")
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Topological Neighbors in Dataset")
        st.write("These labelled profiles in the dataset share the strongest topological proximity with the tested account:")
        neighbor_table = motif_data["neighbor_df"].copy()
        st.dataframe(neighbor_table, use_container_width=True)
    else:
        st.info("👆 Paste an Instagram URL above or click a quick sample (e.g. Cristiano, NASA, or Sample Spam Bot) to see instant detection results!")


# ==============================================================================
# TAB 2: Model Performance
# ==============================================================================
with tab2:
    st.subheader("🏆 Multi-Model Benchmark & Leaderboard")
    st.write("Systematic comparison of 11 diverse algorithms across key evaluation metrics on fake profile detection:")
    try:
        bm_df = pd.read_csv("benchmark_summary.csv")
        display_cols = ["Rank", "Model", "Type", "Accuracy", "F1_Score", "Precision", "Recall", "ROC_AUC", "Specificity", "Train_Time_s"]
        st.dataframe(
            bm_df[display_cols].style.format({
                "Accuracy": "{:.1%}",
                "F1_Score": "{:.3f}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "ROC_AUC": "{:.4f}",
                "Specificity": "{:.1%}",
                "Train_Time_s": "{:.3f}s"
            }),
            use_container_width=True
        )

        fig_bm, ax_bm = plt.subplots(figsize=(10, 4.2))
        models_plot = bm_df["Model"].tolist()[::-1]
        f1_vals = bm_df["F1_Score"].tolist()[::-1]
        acc_vals = bm_df["Accuracy"].tolist()[::-1]
        y_pos = np.arange(len(models_plot))
        ax_bm.barh(y_pos - 0.2, acc_vals, height=0.35, label="Accuracy", color="#3b82f6")
        ax_bm.barh(y_pos + 0.2, f1_vals, height=0.35, label="F1 Score", color="#10b981")
        ax_bm.set_yticks(y_pos)
        ax_bm.set_yticklabels(models_plot)
        ax_bm.set_xlim(0.7, 1.02)
        ax_bm.set_xlabel("Score")
        ax_bm.set_title("Model Accuracy & F1-Score Comparison")
        ax_bm.legend(loc="lower right")
        ax_bm.grid(axis='x', alpha=0.3)
        fig_bm.tight_layout()
        st.pyplot(fig_bm)
        plt.close(fig_bm)
    except Exception as e:
        st.info("Run `python benchmark_engine.py` to regenerate the multi-model benchmark.")

    st.divider()
    st.subheader("Interactive GNN Training & Convergence")
    result = st.session_state.model_result
    if result is None:
        st.info("No interactive GNN model has been trained yet. Click **🚀 Analyze Profile Link** or a sample on Tab 1 to run training.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']:.3f}")
        m2.metric("Precision", f"{result['precision']:.3f}")
        m3.metric("Recall", f"{result['recall']:.3f}")
        m4.metric("F1-Score", f"{result['f1']:.3f}")

        st.divider()
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"**Training Loss Progression ({result.get('model_type', 'GAT')})**")
            st.caption("Includes ReduceLROnPlateau learning rate decays.")
            fig, ax = plt.subplots()
            ax.plot(result["losses"], color="#2563eb", linewidth=2, label="Cross-Entropy Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.25)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with c_right:
            st.markdown("**Confusion Matrix Heatmap**")
            st.caption("Evaluated on stratified hold-out test set.")
            fig, ax = plt.subplots()
            matrix = result["confusion_matrix"]
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Pred Real", "Pred Fake"])
            ax.set_yticks([0, 1], labels=["Actual Real", "Actual Fake"])
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(result["classification_report"]).T.round(3), use_container_width=True)


# ==============================================================================
# TAB 3: Dataset Overview
# ==============================================================================
with tab3:
    st.subheader("Dataset Statistics & Previews")
    c1, c2, c3 = st.columns(3)
    c1.metric("Profiles Sampled", f"{len(df):,}")
    c2.metric("Features Analyzed", len(feature_cols))
    c3.metric("Fake Accounts", f"{int(df['fake'].sum()):,} (50.0%)")
    st.dataframe(df.head(15), use_container_width=True)
    st.caption("Data source: Instagram.csv containing 5,000 balanced user profiles.")


# ==============================================================================
# TAB 4: SpaCEy Research & Methods
# ==============================================================================
with tab4:
    st.subheader("🧬 SpaCEy Methods Integration (Saez Lab)")
    st.markdown(r"""
    This project integrates methodologies from **SpaCEy (Spatial Clinical Explainability)**, an explainable 
    Graph Neural Network framework developed by the **Saez Lab (Julio Saez-Rodriguez group)** and published in *Nature Communications*.

    ### 1. Graph Attention Network (GAT) vs Fixed GCN
    - **In standard GCN**: Message passing uses static degree normalization:
      $$\hat{A} = D^{-1/2} A D^{-1/2}$$
      Every neighbor is weighted purely based on graph degrees, regardless of feature content.
    - **In GAT (SpaCEy Inspired)**: The network computes parameterized attention coefficients $\alpha_{ij}$:
      $$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [W h_i \,\|\, W h_j]\right)$$
      $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
      This allows the model to learn which similar accounts are **true signals of automated bot behavior** versus harmless coincidences.

    ### 2. GNNExplainer: Mutual Information & Mask Optimization
    - SpaCEy implements GNNExplainer to identify compact subgraphs and salient subsets of features.
    - The objective jointly optimizes:
      1. **Feature Mask** $m_{feat} \in [0, 1]^F$
      2. **Edge / Neighbor Mask** $m_{edge} \in [0, 1]^K$
    - Mathematical formulation:
      $$\min_{m_{feat}, m_{edge}} \mathcal{L}_{CE}(Y, \hat{Y}) + \lambda_1 \|m_{feat}\|_1 + \lambda_2 \mathcal{H}(m_{feat}) + \lambda_3 \|m_{edge}\|_1 + \lambda_4 \mathcal{H}(m_{edge})$$
    - The entropy penalty $\mathcal{H}(m)$ enforces decisive binary-like mask selections, isolating the key factors.

    ### 3. Topological Motif & Syndicate Discovery
    - SpaCEy discovers functional tissue motifs by analyzing local graph communities without predefined cell types.
    - We adapt this to detect **Bot Syndicates / Sybil Clusters**:
      - If the user account embeds in a neighborhood with high fake profile density and high edge influence, it is categorized as a **Bot Farm Motif**.
      - If it embeds in organic human clusters, it is recognized as an **Authentic Community Motif**.

    ### 4. ReduceLROnPlateau Learning Rate Scheduling
    - Implements adaptive learning rate decay (`factor=0.5, patience=5`) to prevent overfitting and guarantee smooth loss convergence.
    """)

st.divider()
st.caption("Academic prototype • Research-inspired by Saez Lab's SpaCEy • Not an official Instagram tool.")
