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

# Traditional ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Import existing GNN pipeline from ml_model
from ml_model import build_graph, GAT, GCN

def prepare_limfadd(file_path):
    """Clean and preprocess LIMFADD dataset."""
    df = pd.read_csv(file_path)
    # Replace Excel division error strings
    df = df.replace(["#DIV/0!", "#VALUE!", "#REF!", "#NUM!", "#N/A"], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Target column is 'Labels'
    # Map: 'Real' -> 0, 'Bot'/'Scam'/'Spam' -> 1
    y_raw = df['Labels'].astype(str).str.strip().str.lower()
    y = y_raw.apply(lambda v: 0 if v == 'real' else 1).to_numpy(dtype=np.int64)
    
    # Feature columns
    feature_cols = [c for c in df.columns if c != 'Labels']
    clean_df = pd.DataFrame()
    
    binary_map = {'yes': 1.0, 'y': 1.0, 'true': 1.0, '1': 1.0, 'no': 0.0, 'n': 0.0, 'false': 0.0, '0': 0.0}
    
    for col in feature_cols:
        col_str = df[col].astype(str).str.strip().str.lower()
        # Check if column is categorical boolean (Bio, Profile Picture, External Link, Threads)
        if set(col_str.dropna().unique()).issubset({'yes', 'no', 'y', 'n', 'true', 'false', '1', '0', 'nan'}):
            clean_df[col] = col_str.map(binary_map).fillna(0.0)
        else:
            col_num = pd.to_numeric(df[col], errors='coerce')
            clean_df[col] = col_num.fillna(col_num.median()).fillna(0.0)
            
    X = clean_df.to_numpy(dtype=np.float32)
    return X, y, list(clean_df.columns), df

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

def run_benchmark():
    file_path = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets\LIMFADD.csv"
    X_raw, y, feature_names, raw_df = prepare_limfadd(file_path)
    print(f"Loaded LIMFADD: {len(X_raw)} samples, {len(feature_names)} features.", flush=True)
    print(f"Target distribution: {np.sum(y == 0)} Real (25%), {np.sum(y == 1)} Fake (75%)", flush=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    
    random_state = 42
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)),
        test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 1. Benchmark 11 Models
    print("\n--- Running 11 Model Benchmark on LIMFADD.csv ---", flush=True)
    models = {
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state, max_iter=100),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=random_state),
        "k-Nearest Neighbors (k-NN)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=random_state),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(LinearSVC(dual=False, random_state=random_state)),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=random_state),
        "Gaussian Naive Bayes": GaussianNB(),
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
            
        metrics = evaluate_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        metrics["Type"] = "Tabular ML"
        metrics["Train_Time_s"] = round(train_time, 3)
        metrics["Infer_Time_ms"] = round(infer_time, 3)
        results.append(metrics)
        print(f"  [+] {name:<30} -> Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1_Score']:.4f} | ROC-AUC: {metrics['ROC_AUC']} ({train_time:.2f}s)", flush=True)
        
    # Evaluate GNN models with baseline k=6
    print("\nTraining GNN baselines (k=6)...", flush=True)
    k_base = 6
    # To keep GNN fast and responsive on CPU with 15k nodes, use 5k stratified subset or all
    # Let's test on 6000 nodes for graph to be ultra-fast
    gnn_sub_idx, _ = train_test_split(np.arange(len(y)), train_size=6000, stratify=y, random_state=random_state)
    X_gnn = X_scaled[gnn_sub_idx]
    y_gnn = y[gnn_sub_idx]
    
    idx_tr_gnn, idx_te_gnn = train_test_split(np.arange(len(y_gnn)), test_size=0.2, stratify=y_gnn, random_state=random_state)
    adj_tensor, adj_mat, edge_index = build_graph(X_gnn, k=k_base)
    
    X_tensor = torch.tensor(X_gnn, dtype=torch.float32)
    y_tensor = torch.tensor(y_gnn, dtype=torch.long)
    train_tensor = torch.tensor(idx_tr_gnn, dtype=torch.long)
    test_tensor = torch.tensor(idx_te_gnn, dtype=torch.long)
    
    for gnn_name in ["GAT (Graph Attention Network)", "GCN (Graph Convolutional)"]:
        is_gat = "GAT" in gnn_name
        torch.manual_seed(random_state)
        model = GAT(X_gnn.shape[1], hidden_dim=32) if is_gat else GCN(X_gnn.shape[1], hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        t0 = time.time()
        for epoch in range(40):
            model.train()
            optimizer.zero_grad()
            logits = model(X_tensor, edge_index=edge_index) if is_gat else model(X_tensor, adjacency=adj_tensor)
            loss = F.cross_entropy(logits[train_tensor], y_tensor[train_tensor])
            loss.backward()
            optimizer.step()
        train_time = time.time() - t0
        
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor, edge_index=edge_index) if is_gat else model(X_tensor, adjacency=adj_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        test_probs = probs[idx_te_gnn, 1]
        test_preds = probs[idx_te_gnn].argmax(axis=1)
        metrics = evaluate_metrics(y_gnn[idx_te_gnn], test_preds, test_probs)
        metrics["Model"] = gnn_name
        metrics["Type"] = "Graph Neural Network (SpaCEy)"
        metrics["Train_Time_s"] = round(train_time, 3)
        metrics["Infer_Time_ms"] = round(train_time / 40 * 1000 / len(idx_te_gnn), 3)
        results.append(metrics)
        print(f"  [+] {gnn_name:<30} -> Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1_Score']:.4f} | ROC-AUC: {metrics['ROC_AUC']} ({train_time:.2f}s)", flush=True)

    df_models = pd.DataFrame(results).sort_values(by=["F1_Score", "Accuracy"], ascending=False).reset_index(drop=True)
    df_models["Rank"] = range(1, len(df_models) + 1)
    
    # Save benchmark
    out_bm_json = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets\limfadd_benchmark_results.json"
    df_models.to_json(out_bm_json, orient="records", indent=2)
    print(f"\n[OK] Model Benchmark saved to: {out_bm_json}", flush=True)
    
    # 2. Systematically Sweep K: k in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    print("\n========================================================", flush=True)
    print("--- Running K-Sensitivity Sweep across K in [2..20] ---", flush=True)
    print("========================================================", flush=True)
    
    k_values = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    k_results = []
    
    for k in k_values:
        print(f"\nEvaluating k = {k}:", flush=True)
        # 2a. k-NN
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        pred_knn = knn.predict(X_test)
        prob_knn = knn.predict_proba(X_test)[:, 1]
        m_knn = evaluate_metrics(y_test, pred_knn, prob_knn)
        k_results.append({
            "k": k, "Algorithm": "k-Nearest Neighbors (k-NN)",
            "Accuracy": m_knn["Accuracy"], "F1_Score": m_knn["F1_Score"],
            "Precision": m_knn["Precision"], "Recall": m_knn["Recall"],
            "ROC_AUC": m_knn["ROC_AUC"], "Specificity": m_knn["Specificity"]
        })
        print(f"  k-NN  (k={k:>2}) -> Acc: {m_knn['Accuracy']:.4f} | F1: {m_knn['F1_Score']:.4f} | ROC-AUC: {m_knn['ROC_AUC']}", flush=True)
        
        # 2b. GAT
        adj_t, _, e_idx = build_graph(X_gnn, k=k)
        torch.manual_seed(random_state)
        gat_model = GAT(X_gnn.shape[1], hidden_dim=32)
        opt_gat = torch.optim.Adam(gat_model.parameters(), lr=0.01, weight_decay=5e-4)
        for ep in range(35):
            gat_model.train()
            opt_gat.zero_grad()
            lgt = gat_model(X_tensor, edge_index=e_idx)
            ls = F.cross_entropy(lgt[train_tensor], y_tensor[train_tensor])
            ls.backward()
            opt_gat.step()
        gat_model.eval()
        with torch.no_grad():
            p_gat = torch.softmax(gat_model(X_tensor, edge_index=e_idx), dim=1).cpu().numpy()
        m_gat = evaluate_metrics(y_gnn[idx_te_gnn], p_gat[idx_te_gnn].argmax(axis=1), p_gat[idx_te_gnn, 1])
        k_results.append({
            "k": k, "Algorithm": "GAT (Graph Attention Network)",
            "Accuracy": m_gat["Accuracy"], "F1_Score": m_gat["F1_Score"],
            "Precision": m_gat["Precision"], "Recall": m_gat["Recall"],
            "ROC_AUC": m_gat["ROC_AUC"], "Specificity": m_gat["Specificity"]
        })
        print(f"  GAT   (k={k:>2}) -> Acc: {m_gat['Accuracy']:.4f} | F1: {m_gat['F1_Score']:.4f} | ROC-AUC: {m_gat['ROC_AUC']}", flush=True)
        
        # 2c. GCN
        torch.manual_seed(random_state)
        gcn_model = GCN(X_gnn.shape[1], hidden_dim=32)
        opt_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)
        for ep in range(35):
            gcn_model.train()
            opt_gcn.zero_grad()
            lgt = gcn_model(X_tensor, adjacency=adj_t)
            ls = F.cross_entropy(lgt[train_tensor], y_tensor[train_tensor])
            ls.backward()
            opt_gcn.step()
        gcn_model.eval()
        with torch.no_grad():
            p_gcn = torch.softmax(gcn_model(X_tensor, adjacency=adj_t), dim=1).cpu().numpy()
        m_gcn = evaluate_metrics(y_gnn[idx_te_gnn], p_gcn[idx_te_gnn].argmax(axis=1), p_gcn[idx_te_gnn, 1])
        k_results.append({
            "k": k, "Algorithm": "GCN (Graph Convolutional)",
            "Accuracy": m_gcn["Accuracy"], "F1_Score": m_gcn["F1_Score"],
            "Precision": m_gcn["Precision"], "Recall": m_gcn["Recall"],
            "ROC_AUC": m_gcn["ROC_AUC"], "Specificity": m_gcn["Specificity"]
        })
        print(f"  GCN   (k={k:>2}) -> Acc: {m_gcn['Accuracy']:.4f} | F1: {m_gcn['F1_Score']:.4f} | ROC-AUC: {m_gcn['ROC_AUC']}", flush=True)

    df_k = pd.DataFrame(k_results)
    out_k_json = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets\limfadd_k_sweep_results.json"
    df_k.to_json(out_k_json, orient="records", indent=2)
    print(f"\n[OK] K-Sweep results saved to: {out_k_json}", flush=True)
    
if __name__ == "__main__":
    run_benchmark()
