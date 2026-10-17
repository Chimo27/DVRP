"""
Encodeur GNN — Graph Attention Network (PyTorch Geometric).

POURQUOI UN VRAI GNN, ET PAS UN TRANSFORMER
-------------------------------------------
L'auto-attention classique (nn.TransformerEncoder) sur un ensemble de nœuds
est mathématiquement un GAT sur graphe complet — mais elle ne voit que les
COORDONNÉES des nœuds : le réseau doit ré-apprendre indirectement la notion
de distance à partir de (x, y).

Ici on utilise GATv2Conv avec `edge_dim=1` : la distance entre chaque paire
de nœuds est injectée comme FEATURE D'ARÊTE dans le calcul du poids
d'attention. Le lien entre deux nœuds dépend donc explicitement de leur
distance — l'information structurelle centrale du problème de tournées.

Référence : Brody, Alon & Yahav (2021), "How Attentive are Graph Attention
Networks?" (GATv2).

GRAPHE
    Complet (toute paire de nœuds reliée), sans self-loops : le véhicule
    peut aller de n'importe quel nœud à n'importe quel autre. L'information
    du nœud lui-même circule via les connexions résiduelles.

    Coût en O(N²) arêtes. Pour de très grandes instances, une piste est un
    graphe creux (k plus proches voisins) — non implémenté ici.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


def build_graph_batch(node_features, edge_weights, node_mask=None):
    """
    Convertit des tenseurs denses batchés en un graphe PyG bloc-diagonal
    (toutes les instances du batch dans un seul grand graphe déconnecté —
    format standard de PyTorch Geometric).

    node_features : (B, N, F)
    edge_weights  : (B, N, N)  — typiquement la matrice de distances

    Retourne x (B*N, F), edge_index (2, E), edge_attr (E, 1)
    """
    B, N, F = node_features.shape
    device = node_features.device

    x = node_features.reshape(B * N, F)

    ar = torch.arange(N, device=device)
    src, dst = torch.meshgrid(ar, ar, indexing="ij")
    keep = src != dst
    src, dst = src[keep], dst[keep]

    offsets = (torch.arange(B, device=device) * N).view(B, 1)
    edge_index = torch.stack([
        (src.unsqueeze(0) + offsets).reshape(-1),
        (dst.unsqueeze(0) + offsets).reshape(-1),
    ], dim=0)

    edge_attr = edge_weights[:, src, dst].reshape(-1, 1)
    if node_mask is not None:
        # [révision] seules les arêtes entre deux nœuds visibles sont conservées :
        # un nœud non révélé n'envoie ni ne reçoit de message.
        ok = (node_mask[:, src] & node_mask[:, dst]).reshape(-1)
        edge_index = edge_index[:, ok]
        edge_attr = edge_attr[ok]
    return x, edge_index, edge_attr


class GNNEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=128, n_layers=3, heads=8,
                 use_edge_attr=True):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim doit être divisible par heads"
        self.hidden_dim = hidden_dim
        self.use_edge_attr = use_edge_attr

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True,
                       edge_dim=1 if use_edge_attr else None, add_self_loops=False)
            for _ in range(n_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                          nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, node_features, edge_weights, node_mask=None):
        B, N, _ = node_features.shape
        x, edge_index, edge_attr = build_graph_batch(node_features, edge_weights, node_mask)
        if not self.use_edge_attr:
            edge_attr = None

        h = self.input_proj(x)
        for conv, an, ffn, fn_ in zip(self.convs, self.attn_norms, self.ffns, self.ffn_norms):
            h = an(h + (conv(h, edge_index, edge_attr) if edge_attr is not None else conv(h, edge_index)))   # message passing + résiduel
            h = fn_(h + ffn(h))                            # feed-forward + résiduel

        return h.view(B, N, self.hidden_dim)
