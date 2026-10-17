"""
Politique DVRP = encodeur GNN (GATv2) + décodeur pointeur.

La politique ne lit JAMAIS l'état interne du simulateur : elle reçoit
l'observation construite par ``DVRPEnv.observe()``. En régime "online"
(problème dynamique), cette observation ne contient que le dépôt et les
requêtes déjà apparues ; les requêtes futures en sont absentes (attributs,
distances, voisinage, résumé du graphe, comptages).

ENCODAGE ÉVÉNEMENTIEL
    Le graphe observable change à chaque apparition de requêtes. Les
    représentations h_i sont recalculées pour les seules instances du lot
    dont l'ensemble des nœuds visibles a changé depuis la décision
    précédente ; sinon elles sont réutilisées (les attributs des nœuds
    visibles sont constants entre deux apparitions).

DÉCODAGE
    contexte = [moyenne des h_i visibles, h_nœud_courant, capacité restante,
                temps courant, fraction servie des requêtes VISIBLES]
    puis attention masquée sur les h_i : seules les actions admissibles
    (requêtes apparues, non servies, compatibles avec la capacité, ou dépôt)
    reçoivent une probabilité non nulle.

Le régime "clairvoyant" (toutes les requêtes visibles dès t=0) n'est
conservé que comme ablation : il correspond à la campagne initiale.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_CONFIG
from rl.gnn_encoder import GNNEncoder

# x, y, demande, is_dynamic, instant d'apparition, temps de service
NODE_FEATURE_DIM = 6
# capacité restante, temps courant, fraction servie
CONTEXT_EXTRA_DIM = 3

COORD_SCALE = 100.0


class DVRPPolicy(nn.Module):
    def __init__(self, embed_dim=None, n_heads=None, n_layers=None, tanh_clip=None,
                 information="online", use_edge_attr=True, observe_future=None):
        super().__init__()
        embed_dim = embed_dim or MODEL_CONFIG["embed_dim"]
        n_heads = n_heads or MODEL_CONFIG["n_heads"]
        n_layers = n_layers or MODEL_CONFIG["n_gnn_layers"]
        self.tanh_clip = tanh_clip or MODEL_CONFIG["tanh_clip"]
        self.embed_dim = embed_dim

        # Régime d'information attendu de l'environnement (mémorisé avec le
        # modèle pour que l'évaluation utilise le même régime que
        # l'entraînement). observe_future : ancien nom, conservé pour
        # compatibilité (True -> "clairvoyant").
        if observe_future is not None:
            information = "clairvoyant" if observe_future else "online"
        self.information = information
        self.encoder = GNNEncoder(NODE_FEATURE_DIM, embed_dim, n_layers, n_heads,
                                  use_edge_attr=use_edge_attr)
        self.context_proj = nn.Linear(embed_dim * 2 + CONTEXT_EXTRA_DIM, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    # -----------------------------------------------------------------
    def _node_features(self, obs):
        horizon = max(obs["horizon"], 1.0)
        cap = obs["capacity"].view(-1, 1).clamp(min=1e-6)
        return torch.cat([
            obs["loc"] / COORD_SCALE,
            (obs["demand"] / cap).unsqueeze(-1),
            obs["is_dyn"].float().unsqueeze(-1),
            (obs["appear"] / horizon).unsqueeze(-1),
            (obs["service"] / horizon).unsqueeze(-1),
        ], dim=-1)

    def encode(self, obs, rows=None):
        """Encode le graphe observable (tout le lot, ou seulement `rows`)."""
        feats = self._node_features(obs)
        dist, vis = obs["dist"], obs["visible"]
        if rows is not None:
            feats, dist, vis = feats[rows], dist[rows], vis[rows]
        feats = feats * vis.unsqueeze(-1).float()
        return self.encoder(feats, dist / COORD_SCALE, vis)

    def decode_step(self, h, obs):
        B, N, D = h.shape
        idx = torch.arange(B, device=h.device)
        horizon = max(obs["horizon"], 1.0)

        w = obs["visible"].float().unsqueeze(-1)
        graph_emb = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        node_emb = h[idx, obs["cur_node"]]

        cap_frac = (obs["cur_cap"] / obs["capacity"].clamp(min=1e-6)).unsqueeze(-1)
        time_frac = (obs["cur_time"] / horizon).unsqueeze(-1)
        served_frac = (obs["n_served_visible"]
                       / obs["n_visible"].clamp(min=1.0)).unsqueeze(-1)

        context = torch.cat([graph_emb, node_emb, cap_frac, time_frac, served_frac], dim=-1)
        q = self.q_proj(self.context_proj(context)).unsqueeze(1)
        k = self.k_proj(h)

        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / (D ** 0.5)
        logits = self.tanh_clip * torch.tanh(logits)

        # -1e9 plutôt que -inf : évite un NaN dans le calcul d'entropie
        # (exp(-inf) * -inf = 0 * -inf = NaN) tout en annulant en pratique
        # la probabilité des actions invalides.
        return logits.masked_fill(~obs["mask"], -1e9)

    # -----------------------------------------------------------------
    def rollout(self, env, greedy=False):
        """
        Déroule un épisode complet en interaction séquentielle avec
        l'environnement. Retourne (log_prob_totale, récompense, entropie
        moyenne, métriques). self.n_encodings compte les encodages par
        instance (moyenne sur le lot).
        """
        if getattr(env, "information", "online") != self.information:
            raise ValueError(f"politique entraînée en régime {self.information!r}, "
                             f"environnement en régime {env.information!r}")
        env.reset()
        obs = env.observe()
        h = self.encode(obs)
        vis = obs["visible"].clone()
        enc_count = torch.ones(h.shape[0], device=h.device)

        B = h.shape[0]
        dev = h.device
        log_probs = torch.zeros(B, device=dev)
        entropies = torch.zeros(B, device=dev)
        n_decisions = torch.zeros(B, device=dev)
        self.last_actions = []          # actions de chaque pas (trace d'inférence)

        for _ in range(env.max_steps):
            if env.done.all():
                break

            changed = (obs["visible"] != vis).any(dim=1) & ~env.done
            if changed.any():
                rows = changed.nonzero(as_tuple=True)[0]
                h = h.clone()
                h[rows] = self.encode(obs, rows)
                vis = obs["visible"].clone()
                enc_count = enc_count + changed.float()
            logits = self.decode_step(h, obs)
            log_p = F.log_softmax(logits, dim=-1)

            if greedy:
                action = log_p.argmax(dim=-1)
            else:
                action = torch.multinomial(log_p.exp(), 1).squeeze(-1)

            alive = (~env.done).float()
            log_probs = log_probs + log_p[torch.arange(B, device=dev), action] * alive
            entropies = entropies - (log_p.exp() * log_p).sum(-1) * alive
            n_decisions = n_decisions + alive

            self.last_actions.append(torch.where(env.done, torch.full_like(action, -1), action))
            env.step(action)
            obs = env.observe()

        self.n_encodings = float(enc_count.mean())
        avg_entropy = entropies / n_decisions.clamp(min=1)
        return log_probs, env.final_reward(), avg_entropy, env.metrics()
