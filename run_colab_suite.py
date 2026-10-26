import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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


# Step 1: Model Architectures (GCN and GAT)
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


# Step 2: Dataset Ingestion and Preprocessing
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


# Step 3: Model Training and Evaluation
def run_full_suite(csv_path_or_df, dataset_name="Dataset"):
    if isinstance(csv_path_or_df, str):
        raw_df = pd.read_csv(csv_path_or_df)
    else:
        raw_df = csv_path_or_df
        
    X_raw, y_raw, feature_names, target_name = prepare_any_dataset(raw_df)
    print(f"\nDataset: {dataset_name}")
    print(f"Total Samples: {len(X_raw):,} | Features: {len(feature_names)}")
    print(f"Target: {target_name} (Real: {np.sum(y_raw==0)}, Fake: {np.sum(y_raw==1)})\n")
    
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
            
        m = evaluate_metrics(y_test, y_pred, y_prob)
        m["Model"] = name
        m["Category"] = "Traditional ML" if name in ["Random Forest (RF)", "Decision Tree (DT)", "Support Vector Machine (SVM)", "Logistic Regression (LR)", "K-Nearest Neighbors (KNN)", "Naive Bayes (Gaussian NB)"] else (
            "Ensemble Learning" if "Boosting" in name or "AdaBoost" in name or "Bagged" in name or "Voting" in name else "Deep Learning"
        )
        m["Train_Time_s"] = round(train_time, 3)
        m["Infer_Time_ms"] = round(infer_time, 3)
        results.append(m)
        print(f"  {name:<35} -> Acc: {m['Accuracy']:.4f} | F1: {m['F1_Score']:.4f} | AUC: {m['ROC_AUC']:.4f}")

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
        print(f"  {g_name:<35} -> Acc: {m['Accuracy']:.4f} | F1: {m['F1_Score']:.4f} | AUC: {m['ROC_AUC']:.4f}")

    df_benchmark = pd.DataFrame(results).sort_values(by=["F1_Score", "Accuracy"], ascending=False).reset_index(drop=True)
    df_benchmark["Rank"] = range(1, len(df_benchmark) + 1)
    return df_benchmark, curve_data


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "Instagram.csv")
    df_benchmark, _ = run_full_suite(csv_file, dataset_name=os.path.basename(csv_file))
    print("\n" + "="*95)
    print("FINAL BENCHMARK LEADERBOARD")
    print("="*95)
    display_cols = ["Rank", "Model", "Category", "Accuracy", "F1_Score", "Precision", "Recall", "ROC_AUC", "Specificity", "Balanced_Accuracy", "Train_Time_s"]
    print(df_benchmark[display_cols].to_string(index=False))
