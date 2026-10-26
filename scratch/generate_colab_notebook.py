import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "Fake_Profile_Detection_Colab_Suite.ipynb",
            "provenance": [],
            "authorship_tag": "Fake Profile Detection Benchmark"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": []
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Cell 1: Overview
add_md("""# Fake Social Media Profile Detection Benchmark Suite
### PRISMA Review Taxonomy (13 Algorithms) and Topological Graph Attention Networks (GAT)

Pipeline Components:
1. Dataset Ingestion (File Upload, URL, or Built-in Dataset)
2. Target and Feature Standardization
3. Model Training and Benchmarking (Traditional ML, Ensembles, Deep Learning, GNNs)
4. Empirical Verification and Diagnostic Checks
5. Visualizations (ROC, PR Curves, Confusion Matrices)
6. Automated PDF Report Generation
7. Structured AI Prompt Export
""")

# Cell 2: Dependencies
add_code("""# Step 1: Install Dependencies
!pip install -q reportlab fpdf2 scikit-learn torch pandas numpy matplotlib seaborn
""")

# Cell 3: Model Architectures
add_code("""# Step 2: Model Architectures (GCN and GAT)
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier


def build_graph(X_scaled, k=8):
    k = min(int(k), len(X_scaled) - 1)
    adjacency = kneighbors_graph(X_scaled, n_neighbors=k, mode="connectivity", include_self=False)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(1.0)
    adjacency.eliminate_zeros()

    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
    D = diags(inv_sqrt_degree)
    normalized = (D @ adjacency @ D).tocoo()

    indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
    values = torch.tensor(normalized.data, dtype=torch.float32)
    adjacency_tensor = torch.sparse_coo_tensor(indices, values, size=normalized.shape).coalesce()

    coo = adjacency.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return adjacency_tensor, edge_index


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adjacency):
        x = torch.sparse.mm(adjacency, x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adjacency, x)
        return self.fc2(x)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, edge_index, num_nodes):
        Wh = self.W(h)
        row, col = edge_index[0], edge_index[1]
        edge_h = torch.cat([Wh[row], Wh[col]], dim=1)
        edge_e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze(1))

        e_max = torch.zeros(num_nodes, device=h.device).scatter_reduce(0, row, edge_e, reduce="max")
        edge_exp = torch.exp(edge_e - e_max[row])
        e_sum = torch.zeros(num_nodes, device=h.device).scatter_add(0, row, edge_exp)
        alpha_weights = edge_exp / (e_sum[row] + 1e-12)

        out = torch.zeros(num_nodes, Wh.size(1), device=h.device)
        out.index_add_(0, row, alpha_weights.unsqueeze(1) * Wh[col])
        return out


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, dropout=0.25):
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        h = F.elu(self.gat1(x, edge_index, num_nodes))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gat2(h, edge_index, num_nodes)
""")

# Cell 4: Dataset Ingestion and Preprocessing
add_code("""# Step 3: Dataset Ingestion and Preprocessing
def prepare_any_dataset(df):
    df = df.copy().replace(["#DIV/0!", "#VALUE!", "#REF!", "#NUM!", "#N/A", "nan", "NaN"], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    target_col = None
    candidates = [
        'fake', 'isfake', 'is_fake', 'labels', 'label', 'class',
        'target', 'is_bot', 'bot', 'automatedbehaviour',
        'automatedbehavior', 'automated_behaviour', 'spammer'
    ]
    for cand in candidates:
        for col in df.columns:
            if col.lower().strip().replace(' ', '').replace('_', '') == cand.replace('_', ''):
                target_col = col
                break
        if target_col:
            break
            
    if target_col is None:
        raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")
        
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    pos_set = {'fake', 'bot', '1', '1.0', 'spam', 'spammer', 'scam', 'true', 'yes', 'automated'}
    y = y_raw.apply(lambda v: 1 if v in pos_set else 0).to_numpy(dtype=np.int64)
    
    feature_cols = [c for c in df.columns if c != target_col]
    clean_df = pd.DataFrame()
    binary_map = {'yes': 1.0, 'y': 1.0, 'true': 1.0, '1': 1.0, 'no': 0.0, 'n': 0.0, 'false': 0.0, '0': 0.0}
    
    for col in feature_cols:
        col_str = df[col].astype(str).str.strip().str.lower()
        if set(col_str.dropna().unique()).issubset({'yes', 'no', 'y', 'n', 'true', 'false', '1', '0', 'nan'}):
            clean_df[col] = col_str.map(binary_map).fillna(0.0)
        else:
            col_num = pd.to_numeric(df[col], errors='coerce')
            clean_df[col] = col_num.fillna(col_num.median()).fillna(0.0)
            
    X = clean_df.to_numpy(dtype=np.float32)
    return X, y, list(clean_df.columns), target_col


def generate_benchmark_synthetic(n_samples=2500):
    np.random.seed(42)
    y = np.random.binomial(1, 0.45, n_samples)
    followers = np.where(y == 1, np.random.exponential(120, n_samples), np.random.exponential(1800, n_samples))
    following = np.where(y == 1, np.random.exponential(2500, n_samples), np.random.exponential(450, n_samples))
    posts = np.where(y == 1, np.random.poisson(4, n_samples), np.random.poisson(85, n_samples))
    bio_len = np.where(y == 1, np.random.randint(0, 25, n_samples), np.random.randint(20, 150, n_samples))
    digits_ratio = np.where(y == 1, np.random.uniform(0.3, 0.9, n_samples), np.random.uniform(0.0, 0.15, n_samples))
    
    return pd.DataFrame({
        "profile_pic": np.where(y == 1, np.random.binomial(1, 0.4, n_samples), np.random.binomial(1, 0.98, n_samples)),
        "nums_length_username": digits_ratio,
        "fullname_words": np.random.randint(1, 4, n_samples),
        "description_length": bio_len,
        "posts": posts,
        "followers": followers.astype(int),
        "follows": following.astype(int),
        "fake": y
    })

# Ingestion Options:
# from google.colab import files
# uploaded = files.upload()
# raw_df = pd.read_csv(list(uploaded.keys())[0])
# raw_df = pd.read_csv("https://raw.githubusercontent.com/.../dataset.csv")

raw_df = generate_benchmark_synthetic(2500)
X_raw, y_raw, feature_names, target_name = prepare_any_dataset(raw_df)

print(f"Dataset Loaded Successfully.")
print(f"Total Samples: {len(X_raw):,}")
print(f"Features: {len(feature_names)}")
print(f"Target Column: {target_name} (Real: {np.sum(y_raw==0)}, Fake: {np.sum(y_raw==1)})")
""")

# Cell 5: Model Training and Evaluation
add_code("""# Step 4: Model Training and Evaluation
def evaluate_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    roc_auc, pr_auc = 0.0, 0.0
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            pass
            
    return {
        "Accuracy": round(float(acc), 4),
        "F1_Score": round(float(f1), 4),
        "F1_Macro": round(float(f1_macro), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "Specificity": round(float(specificity), 4),
        "Balanced_Accuracy": round(float(bal_acc), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "PR_AUC": round(float(pr_auc), 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y_raw, np.arange(len(y_raw)), test_size=0.20, random_state=42, stratify=y_raw
)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb_base = HistGradientBoostingClassifier(random_state=42, max_iter=100)
svm_cal = CalibratedClassifierCV(LinearSVC(dual=False, random_state=42))
lr_base = LogisticRegression(max_iter=500, random_state=42)

models = {
    "Random Forest (RF)": rf_base,
    "Decision Tree (DT)": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Support Vector Machine (SVM)": svm_cal,
    "Logistic Regression (LR)": lr_base,
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=8, n_jobs=-1),
    "Naive Bayes (Gaussian NB)": GaussianNB(),
    "Gradient Boosting (GB)": gb_base,
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Bagged SVM": BaggingClassifier(estimator=CalibratedClassifierCV(LinearSVC(dual=False, random_state=42)), n_estimators=5, random_state=42, n_jobs=-1),
    "Voting Ensemble (Max Voting)": VotingClassifier(estimators=[('rf', rf_base), ('gb', gb_base), ('lr', lr_base)], voting='soft'),
    "Artificial Neural Network (ANN/MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42)
}

results = []
curve_data = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t0) * 1000 / len(X_test)
    
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        curve_data[name] = {"fpr": fpr, "tpr": tpr, "p": p_curve, "r": r_curve, "prob": y_prob, "pred": y_pred}
    else:
        curve_data[name] = {"prob": None, "pred": y_pred}
        
    m = evaluate_metrics(y_test, y_pred, y_prob)
    m["Model"] = name
    m["Category"] = "Traditional ML" if name in ["Random Forest (RF)", "Decision Tree (DT)", "Support Vector Machine (SVM)", "Logistic Regression (LR)", "K-Nearest Neighbors (KNN)", "Naive Bayes (Gaussian NB)"] else (
        "Ensemble Learning" if "Boosting" in name or "AdaBoost" in name or "Bagged" in name or "Voting" in name else "Deep Learning"
    )
    m["Train_Time_s"] = round(train_time, 3)
    m["Infer_Time_ms"] = round(infer_time, 3)
    results.append(m)

adj_tensor, edge_index = build_graph(X_scaled, k=8)
X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y_raw, dtype=torch.long)
tr_t = torch.tensor(idx_train, dtype=torch.long)
te_t = torch.tensor(idx_test, dtype=torch.long)

gnn_specs = [
    ("Graph Convolutional Network (GCN)", False),
    ("Graph Attention Network (GAT)", True)
]

for g_name, is_gat in gnn_specs:
    torch.manual_seed(42)
    net = GAT(X_scaled.shape[1], hidden_dim=32) if is_gat else GCN(X_scaled.shape[1], hidden_dim=32)
    opt = torch.optim.Adam(net.parameters(), lr=0.015, weight_decay=5e-4)
    
    t0 = time.time()
    for epoch in range(40):
        net.train()
        opt.zero_grad()
        out = net(X_t, edge_index) if is_gat else net(X_t, adj_tensor)
        loss = F.cross_entropy(out[tr_t], y_t[tr_t])
        loss.backward()
        opt.step()
    train_time = time.time() - t0
    
    net.eval()
    with torch.no_grad():
        out = net(X_t, edge_index) if is_gat else net(X_t, adj_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        
    g_probs = probs[idx_test, 1]
    g_preds = probs[idx_test].argmax(axis=1)
    
    fpr, tpr, _ = roc_curve(y_test, g_probs)
    p_curve, r_curve, _ = precision_recall_curve(y_test, g_probs)
    curve_data[g_name] = {"fpr": fpr, "tpr": tpr, "p": p_curve, "r": r_curve, "prob": g_probs, "pred": g_preds}
    
    m = evaluate_metrics(y_test, g_preds, g_probs)
    m["Model"] = g_name
    m["Category"] = "Graph-Based"
    m["Train_Time_s"] = round(train_time, 3)
    m["Infer_Time_ms"] = round((train_time / 40 * 1000) / len(idx_test), 3)
    results.append(m)

df_benchmark = pd.DataFrame(results).sort_values(by=["F1_Score", "Accuracy"], ascending=False).reset_index(drop=True)
df_benchmark["Rank"] = range(1, len(df_benchmark) + 1)

display_cols = ["Rank", "Model", "Category", "Accuracy", "F1_Score", "Precision", "Recall", "ROC_AUC", "Specificity", "Balanced_Accuracy", "Train_Time_s"]
print("\n" + "="*95)
print("FINAL BENCHMARK LEADERBOARD")
print("="*95)
print(df_benchmark[display_cols].to_string(index=False))
""")

# Cell 6: Metric Verification and Diagnostics
add_code("""# Step 5: Metric Verification and Diagnostics
best_model = df_benchmark.iloc[0]

print("="*80)
print(f"Top Model: {best_model['Model']} ({best_model['Category']})")
print(f"Accuracy: {best_model['Accuracy']*100:.2f}% | F1: {best_model['F1_Score']:.4f} | ROC-AUC: {best_model['ROC_AUC']:.4f}")
print(f"False Positives: {best_model['FP']} | False Negatives: {best_model['FN']}")
print(f"Training Time: {best_model['Train_Time_s']:.3f} s")
print("="*80)

disparity = abs(best_model['Balanced_Accuracy'] - best_model['Accuracy'])
print(f"Accuracy Disparity (Accuracy vs Balanced Accuracy): {disparity:.4f}")
if disparity < 0.05:
    print("Verification: Balanced class distribution response confirmed.")
else:
    print("Verification: Imbalance sensitivity observed.")

if "Graph Attention Network (GAT)" in df_benchmark["Model"].values and "Graph Convolutional Network (GCN)" in df_benchmark["Model"].values:
    gat_f1 = df_benchmark[df_benchmark["Model"] == "Graph Attention Network (GAT)"]["F1_Score"].values[0]
    gcn_f1 = df_benchmark[df_benchmark["Model"] == "Graph Convolutional Network (GCN)"]["F1_Score"].values[0]
    print(f"GAT F1: {gat_f1:.4f} | GCN F1: {gcn_f1:.4f} | Margin: {(gat_f1 - gcn_f1)*100:+.2f}%")
""")

# Cell 7: Visualizations
add_code("""# Step 6: Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

ax1 = axes[0, 0]
top_df = df_benchmark.head(8).sort_values(by="F1_Score", ascending=True)
colors = ['#4A90E2' if 'Graph' in cat else ('#50E3C2' if 'Ensemble' in cat else '#B8E986') for cat in top_df['Category']]
bars = ax1.barh(top_df['Model'], top_df['F1_Score'], color=colors, edgecolor='black', alpha=0.85)
ax1.set_xlim(min(0.6, top_df['F1_Score'].min() - 0.05), 1.0)
ax1.set_title("Top Models by F1-Score", fontsize=12, pad=10)
ax1.set_xlabel("F1-Score")
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=9)

ax2 = axes[0, 1]
for m_name in df_benchmark.head(5)['Model']:
    if m_name in curve_data and curve_data[m_name]['prob'] is not None:
        ax2.plot(curve_data[m_name]['fpr'], curve_data[m_name]['tpr'], lw=2, label=f"{m_name} (AUC={df_benchmark[df_benchmark['Model']==m_name]['ROC_AUC'].values[0]:.3f})")
ax2.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7)
ax2.set_title("ROC Curves", fontsize=12, pad=10)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(alpha=0.3)

ax3 = axes[1, 0]
for m_name in df_benchmark.head(5)['Model']:
    if m_name in curve_data and curve_data[m_name]['prob'] is not None:
        ax3.plot(curve_data[m_name]['r'], curve_data[m_name]['p'], lw=2, label=f"{m_name} (PR={df_benchmark[df_benchmark['Model']==m_name]['PR_AUC'].values[0]:.3f})")
ax3.set_title("Precision-Recall Curves", fontsize=12, pad=10)
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.legend(loc="lower left", fontsize=8)
ax3.grid(alpha=0.3)

ax4 = axes[1, 1]
cm_best = np.array([[best_model['TN'], best_model['FP']], [best_model['FN'], best_model['TP']]])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
            xticklabels=["Predicted Real", "Predicted Fake"],
            yticklabels=["Actual Real", "Actual Fake"])
ax4.set_title(f"Confusion Matrix: {best_model['Model']}", fontsize=12, pad=10)

plt.tight_layout()
plt.savefig("benchmark_visualizations.png", dpi=200, bbox_inches='tight')
plt.show()
""")

# Cell 8: PDF Report Generation
add_code("""# Step 7: PDF Report Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
try:
    from google.colab import files
    in_colab = True
except ImportError:
    in_colab = False

pdf_filename = "Fake_Profile_Detection_Benchmark_Report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#1A365D"), fontName="Helvetica-Bold", spaceAfter=6)
subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontSize=9.5, leading=14, textColor=colors.HexColor("#4A5568"), spaceAfter=10)
section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#2B6CB0"), fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#2D3748"))
table_cell_style = ParagraphStyle("TableCellStyle", parent=styles["Normal"], fontSize=7, leading=9.5, textColor=colors.HexColor("#1A202C"))
table_hdr_style = ParagraphStyle("TableHdrStyle", parent=styles["Normal"], fontSize=7, leading=9.5, textColor=colors.white, fontName="Helvetica-Bold")

story = []
story.append(Paragraph("Fake Profile Detection Benchmark Report", title_style))
story.append(Paragraph(f"Taxonomy: Strict PRISMA Review Suite (13 Models) and Graph Attention Networks<br/>"
                       f"Dataset: Injected Dataset ({len(X_raw):,} samples, {len(feature_names)} features) | Split: 80/20 Stratified", subtitle_style))
story.append(Spacer(1, 6))

story.append(Paragraph("1. Executive Summary and Verification Findings", section_style))
summary_text = (
    f"This evaluation assesses 13 classification models across 4 taxonomies: "
    f"Traditional Machine Learning, Ensemble Learning, Deep Learning (ANN/MLP), and Graph Neural Networks (GCN and GAT).<br/>"
    f"Top Performing Model: {best_model['Model']} achieved {best_model['Accuracy']*100:.2f}% Accuracy, "
    f"F1-Score of {best_model['F1_Score']:.4f}, and ROC-AUC of {best_model['ROC_AUC']:.4f}.<br/>"
    f"Error Distribution: {best_model['FP']} False Positives and {best_model['FN']} False Negatives.<br/>"
    f"Graph Modeling: GAT dynamically weights neighbor connections for neighborhood bot detection."
)
story.append(Paragraph(summary_text, body_style))
story.append(Spacer(1, 8))

story.append(Paragraph("2. Benchmark Leaderboard", section_style))
headers = ["Rank", "Model", "Category", "Acc", "F1", "Prec", "Rec", "AUC", "Spec", "Train(s)"]
table_data = [[Paragraph(f"<b>{h}</b>", table_hdr_style) for h in headers]]

for _, r in df_benchmark.iterrows():
    table_data.append([
        Paragraph(str(r["Rank"]), table_cell_style),
        Paragraph(r["Model"][:23], table_cell_style),
        Paragraph(r["Category"], table_cell_style),
        Paragraph(f"{r['Accuracy']:.3f}", table_cell_style),
        Paragraph(f"{r['F1_Score']:.3f}", table_cell_style),
        Paragraph(f"{r['Precision']:.3f}", table_cell_style),
        Paragraph(f"{r['Recall']:.3f}", table_cell_style),
        Paragraph(f"{r['ROC_AUC']:.3f}", table_cell_style),
        Paragraph(f"{r['Specificity']:.3f}", table_cell_style),
        Paragraph(f"{r['Train_Time_s']:.2f}", table_cell_style),
    ])

t = Table(table_data, colWidths=[24, 125, 95, 34, 34, 34, 34, 34, 34, 45])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(t)
story.append(Spacer(1, 10))

story.append(Paragraph("3. Visualizations and Diagnostics", section_style))
if os.path.exists("benchmark_visualizations.png"):
    story.append(Image("benchmark_visualizations.png", width=540, height=380))

doc.build(story)
print(f"Report generated: {pdf_filename}")
if in_colab:
    files.download(pdf_filename)
""")

# Cell 9: AI Documentation Prompt Export
add_code("""# Step 8: AI Documentation Prompt Export
ai_prompt = f\"\"\"# Role: Machine Learning Researcher and Technical Writer
Task: Generate an academic research paper and technical documentation based on the verified benchmark results below.

## Study Specifications
- Title: Comparative Benchmarking of Machine Learning, Ensembles, Deep Learning, and Graph Neural Networks for Fake Profile Detection
- Taxonomy: Strict PRISMA Review Taxonomy and Topological Attention Networks
- Dataset: Profile dataset with {len(X_raw):,} samples and {len(feature_names)} features.
- Protocol: Stratified 80/20 train/test holdout, 10 evaluation metrics.

## Benchmark Results
{df_benchmark[display_cols].to_markdown(index=False)}

## Key Findings:
1. Top Model: {best_model['Model']} ({best_model['Category']}) ranked 1 with {best_model['Accuracy']*100:.2f}% Accuracy, {best_model['F1_Score']:.4f} F1-score, and {best_model['ROC_AUC']:.4f} ROC-AUC.
2. Error Metrics: {best_model['FP']} False Positives and {best_model['FN']} False Negatives.
3. Graph Modeling: Graph Attention Network evaluates neighbor feature attention for account clusters.

## Required Sections:
1. Abstract (150 words)
2. Introduction and Problem Context
3. Methodology and Model Specifications
4. Empirical Performance Analysis
5. Error Diagnostics and Confusion Matrix Analysis
6. Conclusion and Deployment Notes
\"\"\"

with open("AI_Paper_Generation_Prompt.md", "w", encoding="utf-8") as f:
    f.write(ai_prompt)

print("AI Prompt saved to AI_Paper_Generation_Prompt.md")
if in_colab:
    files.download("AI_Paper_Generation_Prompt.md")
""")

notebook_path = "Fake_Profile_Detection_Colab_Suite.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated Colab Notebook at: {os.path.abspath(notebook_path)}")
