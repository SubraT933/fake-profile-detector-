"""
Strict PRISMA Review Algorithm Benchmark for Fake Profile Detection
Based on Sarthak & Sanidhya (2026): "Fake Account Detection on Social Media: A Systematic Review"

Strict Algorithm Suite:
Traditional ML (Section 7.2):
  1. Random Forest (RF)
  2. Decision Tree (DT)
  3. Support Vector Machine (SVM)
  4. Logistic Regression (LR)
  5. K-Nearest Neighbors (KNN)
  6. Naive Bayes (Gaussian NB)
Ensemble Learning (Section 7.3):
  7. Gradient Boosting (GB)
  8. AdaBoost
  9. Bagged SVM
 10. Voting Ensemble (Max Voting)
Deep Learning (Section 7.4):
 11. Artificial Neural Network (ANN / MLP)
Graph-Based Learning (Section 7.6):
 12. Graph Convolutional Network (GCN)
 13. Graph Attention Network (GAT)
"""

import os
import sys

project_root = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix
)

# Traditional ML (Section 7.2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Ensemble Learning (Section 7.3)
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier, # High performance gradient boosting
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier
)

# Deep Learning (Section 7.4)
from sklearn.neural_network import MLPClassifier

# Graph-Based (Section 7.6)
from ml_model import build_graph, GAT, GCN

def prepare_any_dataset(df):
    df = df.copy().replace(["#DIV/0!", "#VALUE!", "#REF!", "#NUM!", "#N/A"], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Target column detection
    target_col = None
    for cand in ['fake', 'labels', 'label', 'class', 'target', 'is_bot', 'bot']:
        for col in df.columns:
            if col.lower().strip() == cand:
                target_col = col
                break
        if target_col:
            break
            
    if target_col is None:
        raise ValueError(f"Target column not found in: {list(df.columns)}")
        
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    pos_set = {'fake', 'bot', '1', 'spam', 'scam', 'true', 'yes'}
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
    
    roc_auc = np.nan
    pr_auc = np.nan
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
        "ROC_AUC": round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
        "PR_AUC": round(float(pr_auc), 4) if not np.isnan(pr_auc) else None,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def run_strict_prisma_benchmark(csv_path, dataset_name, sample_for_gnn=5000):
    print(f"\n========================================================", flush=True)
    print(f"Benchmarking Dataset: {dataset_name}", flush=True)
    print(f"Strict PRISMA Review Taxonomy (Sarthak & Sanidhya, 2026)", flush=True)
    print(f"========================================================", flush=True)
    
    df = pd.read_csv(csv_path)
    X_raw, y, feature_names, target_col = prepare_any_dataset(df)
    print(f"Samples: {len(X_raw)}, Features: {len(feature_names)}, Target: '{target_col}'", flush=True)
    print(f"Class split: {np.sum(y == 0)} Real (0), {np.sum(y == 1)} Fake (1)", flush=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    
    random_state = 42
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Base estimators for ensembles
    rf_base = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    gb_base = HistGradientBoostingClassifier(random_state=random_state, max_iter=100)
    svm_cal = CalibratedClassifierCV(LinearSVC(dual=False, random_state=random_state))
    lr_base = LogisticRegression(max_iter=500, random_state=random_state)
    
    # Strict Paper Suite (Section 7.2 - 7.6)
    models = {
        # Traditional Machine Learning (Section 7.2)
        "Random Forest (RF)": rf_base,
        "Decision Tree (DT)": DecisionTreeClassifier(max_depth=12, random_state=random_state),
        "Support Vector Machine (SVM)": svm_cal,
        "Logistic Regression (LR)": lr_base,
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=8, n_jobs=-1),
        "Naive Bayes (Gaussian NB)": GaussianNB(),
        
        # Ensemble Learning (Section 7.3)
        "Gradient Boosting (GB)": gb_base,
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=random_state),
        "Bagged SVM": BaggingClassifier(estimator=CalibratedClassifierCV(LinearSVC(dual=False, random_state=random_state)), n_estimators=5, random_state=random_state, n_jobs=-1),
        "Voting Ensemble (Max Voting)": VotingClassifier(estimators=[('rf', rf_base), ('gb', gb_base), ('lr', lr_base)], voting='soft'),
        
        # Deep Learning (Section 7.4)
        "Artificial Neural Network (ANN/MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=random_state),
    }
    
    results = []
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
            
        m = evaluate_metrics(y_test, y_pred, y_prob)
        m["Model"] = name
        m["Category"] = "Traditional ML (Sec 7.2)" if "RF" in name or "DT" in name or "SVM" in name and "Bagged" not in name or "LR" in name or "KNN" in name or "NB" in name else (
            "Ensemble Learning (Sec 7.3)" if "Boosting" in name or "AdaBoost" in name or "Bagged" in name or "Voting" in name else "Deep Learning (Sec 7.4)"
        )
        m["Train_Time_s"] = round(train_time, 3)
        m["Infer_Time_ms"] = round(infer_time, 3)
        results.append(m)
        print(f"  [+] {name:<35} -> Acc: {m['Accuracy']:.4f} | F1: {m['F1_Score']:.4f} | ROC-AUC: {m['ROC_AUC']} ({train_time:.2f}s)", flush=True)

    # Graph-Based Approaches (Section 7.6)
    k_opt = 8
    if len(X_scaled) > sample_for_gnn:
        sub_idx, _ = train_test_split(np.arange(len(y)), train_size=sample_for_gnn, stratify=y, random_state=random_state)
        X_gnn, y_gnn = X_scaled[sub_idx], y[sub_idx]
    else:
        X_gnn, y_gnn = X_scaled, y
        
    idx_tr_g, idx_te_g = train_test_split(np.arange(len(y_gnn)), test_size=0.2, stratify=y_gnn, random_state=random_state)
    adj_t, _, e_idx = build_graph(X_gnn, k=k_opt)
    X_t = torch.tensor(X_gnn, dtype=torch.float32)
    y_t = torch.tensor(y_gnn, dtype=torch.long)
    tr_t = torch.tensor(idx_tr_g, dtype=torch.long)
    
    gnn_models = [
        ("Graph Convolutional Network (GCN)", False),
        ("Graph Attention Network (GAT)", True)
    ]
    
    for g_name, is_gat in gnn_models:
        torch.manual_seed(random_state)
        net = GAT(X_gnn.shape[1], hidden_dim=32) if is_gat else GCN(X_gnn.shape[1], hidden_dim=32)
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
        t0 = time.time()
        for ep in range(35):
            net.train()
            opt.zero_grad()
            lgt = net(X_t, edge_index=e_idx) if is_gat else net(X_t, adjacency=adj_t)
            ls = F.cross_entropy(lgt[tr_t], y_t[tr_t])
            ls.backward()
            opt.step()
        train_time = time.time() - t0
        net.eval()
        with torch.no_grad():
            lgt = net(X_t, edge_index=e_idx) if is_gat else net(X_t, adjacency=adj_t)
            probs = torch.softmax(lgt, dim=1).cpu().numpy()
            
        te_probs = probs[idx_te_g, 1]
        te_preds = probs[idx_te_g].argmax(axis=1)
        m = evaluate_metrics(y_gnn[idx_te_g], te_preds, te_probs)
        m["Model"] = g_name
        m["Category"] = "Graph-Based (Sec 7.6)"
        m["Train_Time_s"] = round(train_time, 3)
        m["Infer_Time_ms"] = round(train_time / 35 * 1000 / len(idx_te_g), 3)
        results.append(m)
        print(f"  [+] {g_name:<35} -> Acc: {m['Accuracy']:.4f} | F1: {m['F1_Score']:.4f} | ROC-AUC: {m['ROC_AUC']} ({train_time:.2f}s)", flush=True)

    df_res = pd.DataFrame(results).sort_values(by=["F1_Score", "Accuracy"], ascending=False).reset_index(drop=True)
    df_res["Rank"] = range(1, len(df_res) + 1)
    return df_res

if __name__ == "__main__":
    d1 = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\Instagram.csv"
    d2 = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets\LIMFADD.csv"
    
    res1 = run_strict_prisma_benchmark(d1, "Instagram.csv (5,000 profiles)")
    res2 = run_strict_prisma_benchmark(d2, "LIMFADD.csv (15,000 profiles)")
    
    out_dir = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets"
    res1.to_json(os.path.join(out_dir, "strict_prisma_instagram_results.json"), orient="records", indent=2)
    res2.to_json(os.path.join(out_dir, "strict_prisma_limfadd_results.json"), orient="records", indent=2)
    print("\n[OK] Strict PRISMA evaluation completed successfully for both datasets!")
