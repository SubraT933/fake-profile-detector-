import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.sparse import diags


TARGET = "fake"


def prepare_data(df):
    """Clean the input dataframe and return numeric features and labels."""
    df = df.copy().replace([np.inf, -np.inf], np.nan)

    if TARGET not in df.columns:
        raise ValueError("CSV must contain a 'fake' target column (0 = real, 1 = fake).")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])]

    feature_cols = [c for c in df.columns if c != TARGET]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(
        df[feature_cols].median(numeric_only=True)
    ).fillna(0)

    if len(df) < 20:
        raise ValueError("Please provide at least 20 labelled profiles.")

    return df, feature_cols


def build_graph(X_scaled, k=6):
    """Build a symmetric k-nearest-neighbour feature-similarity graph."""
    k = min(int(k), len(X_scaled) - 1)
    adjacency = kneighbors_graph(
        X_scaled, n_neighbors=k, mode="connectivity", include_self=False
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(1.0)
    adjacency.eliminate_zeros()

    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt_degree = np.power(np.maximum(degree, 1e-12), -0.5)
    D = diags(inv_sqrt_degree)
    normalized = (D @ adjacency @ D).tocoo()

    indices = torch.tensor(
        np.vstack([normalized.row, normalized.col]), dtype=torch.long
    )
    values = torch.tensor(normalized.data, dtype=torch.float32)
    adjacency_tensor = torch.sparse_coo_tensor(
        indices, values, size=normalized.shape
    ).coalesce()

    # Edge index for Graph Attention Networks & GNNExplainer
    coo = adjacency.tocoo()
    edge_index = torch.tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long
    )

    return adjacency_tensor, adjacency, edge_index


# ==============================================================================
# Model Architectures
# ==============================================================================

class GCN(nn.Module):
    """Kipf & Welling Graph Convolutional Network for binary classification."""
    def __init__(self, input_dim, hidden_dim=32, dropout=0.35):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adjacency, edge_index=None):
        x = torch.sparse.mm(adjacency, x)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adjacency, x)
        return self.layer2(x)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with dynamic edge attention coefficients.
    Inspired by Saez Lab's SpaCEy architecture (https://github.com/saezlab/SpaCEy).
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index, num_nodes, edge_weight_mask=None):
        Wh = self.W(h)
        row, col = edge_index[0], edge_index[1]
        edge_h = torch.cat([Wh[row], Wh[col]], dim=1)
        edge_e = self.leakyrelu(torch.matmul(edge_h, self.a).squeeze(1))

        # Numerically stable edge attention softmax
        edge_e_max = torch.zeros(num_nodes, device=h.device).scatter_reduce(
            0, row, edge_e, reduce="amax", include_self=False
        )
        edge_e_exp = torch.exp(edge_e - edge_e_max[row])
        if edge_weight_mask is not None:
            edge_e_exp = edge_e_exp * edge_weight_mask

        edge_e_sum = torch.zeros(num_nodes, device=h.device).scatter_add_(0, row, edge_e_exp)
        alpha = edge_e_exp / (edge_e_sum[row] + 1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        weighted_Wh = Wh[col] * alpha.unsqueeze(1)
        out = torch.zeros(num_nodes, self.out_features, device=h.device).scatter_add_(
            0, row.unsqueeze(1).expand(-1, self.out_features), weighted_Wh
        )
        return out, alpha


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) for inductive and transductive node classification.
    Inspired by Saez Lab's SpaCEy GNN formulation.
    """
    def __init__(self, input_dim, hidden_dim=32, num_classes=2, dropout=0.2):
        super().__init__()
        self.gat1 = GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, num_classes, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adjacency=None, edge_index=None, edge_weight_mask=None, return_attention=False):
        num_nodes = x.size(0)
        h, a1 = self.gat1(x, edge_index, num_nodes, edge_weight_mask=edge_weight_mask)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out, a2 = self.gat2(h, edge_index, num_nodes, edge_weight_mask=edge_weight_mask)
        if return_attention:
            return out, (a1, a2)
        return out


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train_gnn(df, k=6, test_size=0.2, epochs=60, hidden_dim=32, seed=42, model_type="GAT"):
    """
    Train a GNN model (GAT or GCN) with ReduceLROnPlateau learning rate scheduling.
    Inspired by SpaCEy training loops (Saez Lab).
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    clean_df, feature_cols = prepare_data(df)
    X_raw = clean_df[feature_cols].to_numpy(dtype=np.float32)
    y = clean_df[TARGET].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_scaled = np.clip(scaler.fit_transform(X_raw), -10, 10).astype(np.float32)

    adjacency_tensor, adjacency_matrix, edge_index = build_graph(X_scaled, k)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    all_indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        all_indices, test_size=float(test_size),
        random_state=int(seed), stratify=y
    )
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    test_tensor = torch.tensor(test_indices, dtype=torch.long)

    if model_type.upper() == "GAT":
        model = GAT(X_scaled.shape[1], int(hidden_dim))
    else:
        model = GCN(X_scaled.shape[1], int(hidden_dim))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # SpaCEy-inspired learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    losses = []

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        if model_type.upper() == "GAT":
            logits = model(X_tensor, edge_index=edge_index)
        else:
            logits = model(X_tensor, adjacency=adjacency_tensor)

        loss = F.cross_entropy(logits[train_tensor], y_tensor[train_tensor])
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach())
        losses.append(loss_val)
        scheduler.step(loss_val)

    model.eval()
    with torch.no_grad():
        if model_type.upper() == "GAT":
            logits = model(X_tensor, edge_index=edge_index)
        else:
            logits = model(X_tensor, adjacency=adjacency_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    test_predictions = probabilities[test_indices].argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y[test_indices], test_predictions),
        "precision": precision_score(y[test_indices], test_predictions, zero_division=0),
        "recall": recall_score(y[test_indices], test_predictions, zero_division=0),
        "f1": f1_score(y[test_indices], test_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y[test_indices], test_predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y[test_indices], test_predictions,
            labels=[0, 1], target_names=["Real", "Fake"],
            zero_division=0, output_dict=True
        ),
        "test_indices": test_indices,
        "probabilities": probabilities,
        "predictions": probabilities.argmax(axis=1),
        "losses": losses,
        "X_scaled": X_scaled,
        "adjacency_matrix": adjacency_matrix,
        "adjacency_tensor": adjacency_tensor,
        "edge_index": edge_index,
        "scaler": scaler,
        "model": model,
        "model_type": model_type.upper(),
        "X_raw": X_raw,
        "y": y,
        "feature_cols": feature_cols,
        "dataframe": clean_df,
    }
    return metrics


# ==============================================================================
# Explainability & Motif Discovery (SpaCEy Inspired)
# ==============================================================================

def explain_profile_saliency(result, row_index):
    """Gradient-times-input feature attribution."""
    model = result["model"]
    X_scaled = result["X_scaled"]
    feature_cols = result["feature_cols"]
    model_type = result.get("model_type", "GCN")

    x = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    model.eval()
    model.zero_grad()

    if model_type == "GAT":
        logits = model(x, edge_index=result["edge_index"])
    else:
        logits = model(x, adjacency=result["adjacency_tensor"])

    predicted_class = int(logits[row_index].argmax().item())
    logits[row_index, predicted_class].backward()

    gradients = x.grad[row_index].detach().cpu().numpy()
    importances = np.abs(gradients * X_scaled[row_index])

    adjacency = result["adjacency_matrix"]
    neighbours = adjacency.getrow(row_index).indices
    neighbours = neighbours[neighbours != row_index]

    if len(neighbours):
        distances = np.linalg.norm(
            X_scaled[neighbours] - X_scaled[row_index], axis=1
        )
        order = np.argsort(distances)[:5]
        neighbours = neighbours[order]
        distances = distances[order]
    else:
        distances = np.array([])

    return {
        "predicted_class": predicted_class,
        "probability": float(result["probabilities"][row_index, predicted_class]),
        "importances": pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False),
        "neighbours": neighbours,
        "distances": distances,
        "actual_class": int(result["y"][row_index]),
    }


def explain_profile_gnn_explainer(result, user_scaled_vector, nearest_indices, epochs=60, lr=0.04):
    """
    GNNExplainer mask optimization method inspired by SpaCEy (Saez Lab).
    Optimizes a feature mask and edge mask over the local subgraph with mutual information,
    sparsity L1 penalty, and entropy penalty.
    """
    feature_cols = result["feature_cols"]
    num_features = len(feature_cols)
    k_neighbors = len(nearest_indices)

    # Construct star subgraph centered at user profile (node 0) connected to k dataset neighbors (nodes 1..k)
    neighbor_scaled = result["X_scaled"][nearest_indices]
    sub_x = torch.tensor(np.vstack([user_scaled_vector, neighbor_scaled]), dtype=torch.float32)
    num_sub_nodes = sub_x.shape[0]

    # Star edges between node 0 and each neighbor
    src = [0] * k_neighbors + list(range(1, k_neighbors + 1))
    dst = list(range(1, k_neighbors + 1)) + [0] * k_neighbors
    sub_edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Trainable feature mask and edge mask
    feat_mask_param = nn.Parameter(torch.zeros(num_features))
    edge_mask_param = nn.Parameter(torch.zeros(len(src)))

    optimizer = torch.optim.Adam([feat_mask_param, edge_mask_param], lr=lr)

    # Use model in eval mode
    model = result["model"]
    model.eval()

    # Determine baseline predicted class from unmasked forward pass
    with torch.no_grad():
        if result.get("model_type") == "GAT":
            base_logits = model(sub_x, edge_index=sub_edge_index)
        else:
            adj_dense = torch.zeros((num_sub_nodes, num_sub_nodes))
            adj_dense[sub_edge_index[0], sub_edge_index[1]] = 1.0
            for i in range(num_sub_nodes):
                adj_dense[i, i] = 1.0
            base_logits = model(sub_x, adjacency=adj_dense.to_sparse())
        target_class = int(base_logits[0].argmax().item())

    # Optimization loop (SpaCEy GNNExplainer objective)
    for _ in range(epochs):
        optimizer.zero_grad()
        m_feat = torch.sigmoid(feat_mask_param)
        m_edge = torch.sigmoid(edge_mask_param)

        masked_x = sub_x * m_feat.unsqueeze(0)

        if result.get("model_type") == "GAT":
            out_logits = model(masked_x, edge_index=sub_edge_index, edge_weight_mask=m_edge)
        else:
            adj_dense = torch.zeros((num_sub_nodes, num_sub_nodes))
            adj_dense[sub_edge_index[0], sub_edge_index[1]] = m_edge
            for i in range(num_sub_nodes):
                adj_dense[i, i] = 1.0
            out_logits = model(masked_x, adjacency=adj_dense.to_sparse())

        # Prediction retention loss (cross-entropy with target class)
        loss_pred = F.cross_entropy(out_logits[0:1], torch.tensor([target_class]))

        # Sparsity penalty (L1 norm)
        loss_feat_size = torch.mean(m_feat)
        loss_edge_size = torch.mean(m_edge)

        # Entropy penalty: pushes mask values decisively toward 0 or 1
        ent_feat = -torch.mean(m_feat * torch.log(m_feat + 1e-12) + (1.0 - m_feat) * torch.log(1.0 - m_feat + 1e-12))
        ent_edge = -torch.mean(m_edge * torch.log(m_edge + 1e-12) + (1.0 - m_edge) * torch.log(1.0 - m_edge + 1e-12))

        loss = loss_pred + 0.08 * (loss_feat_size + loss_edge_size) + 0.05 * (ent_feat + ent_edge)
        loss.backward()
        optimizer.step()

    final_feat_mask = torch.sigmoid(feat_mask_param).detach().cpu().numpy()
    final_edge_mask = torch.sigmoid(edge_mask_param).detach().cpu().numpy()[:k_neighbors]

    feature_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_feat_mask
    }).sort_values("Importance", ascending=False)

    return feature_df, final_edge_mask


def predict_user_profile(result, profile_values, k=6):
    """Estimate a new profile's class from nearby dataset profiles' model probabilities."""
    row = pd.DataFrame([profile_values], columns=result["feature_cols"])
    row_scaled = result["scaler"].transform(row).astype(np.float32)
    existing = result["X_scaled"]
    distances = np.linalg.norm(existing - row_scaled, axis=1)
    nearest = np.argsort(distances)[:min(int(k), len(distances))]
    weights = 1.0 / (distances[nearest] + 1e-6)
    weights = weights / weights.sum()
    fake_probability = float(np.sum(result["probabilities"][nearest, 1] * weights))
    predicted_class = int(fake_probability >= 0.5)
    return {
        "predicted_class": predicted_class,
        "fake_probability": fake_probability,
        "neighbours": nearest,
        "distances": distances[nearest],
        "weights": weights,
        "row_scaled": row_scaled[0],
    }


def explain_user_profile(result, profile_values, k=6, method="gnn_explainer"):
    """
    Explain user profile using SpaCEy-inspired GNNExplainer (default) or Gradient x Input.
    """
    row = pd.DataFrame([profile_values], columns=result["feature_cols"])
    row_scaled = result["scaler"].transform(row).astype(np.float32)[0]
    existing = result["X_scaled"]
    distances = np.linalg.norm(existing - row_scaled, axis=1)
    nearest = np.argsort(distances)[:min(int(k), len(distances))]

    if method == "gnn_explainer":
        feat_df, edge_weights = explain_profile_gnn_explainer(
            result, row_scaled, nearest, epochs=50, lr=0.04
        )
        return feat_df, edge_weights
    else:
        attributions = []
        for idx in nearest:
            exp = explain_profile_saliency(result, int(idx))
            attributions.append(exp["importances"]["Importance"].to_numpy(dtype=float))

        if not attributions:
            feat_df = pd.DataFrame({
                "Feature": result["feature_cols"],
                "Importance": np.zeros(len(result["feature_cols"]))
            })
        else:
            importance = np.mean(np.vstack(attributions), axis=0)
            feat_df = pd.DataFrame({
                "Feature": result["feature_cols"],
                "Importance": importance
            }).sort_values("Importance", ascending=False)
        return feat_df, np.ones(len(nearest)) / len(nearest)


def analyze_subgraph_motif(result, profile_values, k=6):
    """
    SpaCEy-inspired Motif & Cluster Discovery.
    Identifies whether the profile's local topological neighborhood constitutes:
    - A 'BOT SYNDICATE / BOT FARM MOTIF'
    - An 'AUTHENTIC HUMAN COMMUNITY MOTIF'
    - A 'BORDERLINE MIXED CLUSTER'
    """
    prediction = predict_user_profile(result, profile_values, k=k)
    nearest = prediction["neighbours"]
    distances = prediction["distances"]
    weights = prediction["weights"]

    labels = result["y"][nearest]
    model_preds = result["predictions"][nearest]

    fake_neighbor_count = int(np.sum(labels == 1))
    real_neighbor_count = int(np.sum(labels == 0))
    weighted_fake_affinity = float(np.sum((labels == 1) * weights))

    if weighted_fake_affinity >= 0.65:
        motif_type = "🤖 Bot Syndicate / Farm Motif"
        motif_color = "#ef4444"
        motif_desc = "The profile strongly clusters with known synthetic/bot accounts in feature space. High risk of coordinated network behavior."
    elif weighted_fake_affinity <= 0.35:
        motif_type = "👤 Authentic Human Community Motif"
        motif_color = "#10b981"
        motif_desc = "The profile tightly embeds within organic human user clusters with typical follower-to-following behavior."
    else:
        motif_type = "⚖️ Borderline / Mixed Neighborhood"
        motif_color = "#f59e0b"
        motif_desc = "The profile sits on the boundary between genuine and automated accounts, sharing attributes with both groups."

    neighbor_df = result["dataframe"].iloc[nearest].copy()
    neighbor_df.insert(0, "Neighbor Index", nearest)
    neighbor_df["Distance"] = np.round(distances, 3)
    neighbor_df["Influence Weight"] = np.round(weights, 3)
    neighbor_df["Status"] = ["Fake (Bot)" if l == 1 else "Real (Human)" for l in labels]

    return {
        "motif_type": motif_type,
        "motif_color": motif_color,
        "motif_desc": motif_desc,
        "fake_neighbor_count": fake_neighbor_count,
        "real_neighbor_count": real_neighbor_count,
        "weighted_fake_affinity": weighted_fake_affinity,
        "neighbor_df": neighbor_df,
        "nearest": nearest,
        "weights": weights,
    }
