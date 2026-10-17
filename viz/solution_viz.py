"""
Visualisation des solutions (tournées) et du déroulement de l'entraînement.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from viz.instance_viz import _apply_style, _clean_axes, DEEP, TEAL, MIDNIGHT, ACCENT, GRAY

TOUR_COLORS = ["#065A82", "#1C7293", "#F2A65A", "#7C9885", "#B5651D",
               "#5B6C8F", "#9A6FB0", "#C25B5B"]


@torch.no_grad()
def rollout_single(policy_or_fn, instance, device="cpu", greedy=True):
    """
    Déroule un épisode sur UNE instance et retourne la séquence de nœuds
    visités. Accepte soit une politique (avec .rollout), soit une fonction
    heuristique de baselines.py.
    """
    from rl.env import DVRPEnv

    env = DVRPEnv([instance], device=device)
    state = env.reset()
    route = [0]

    is_nn_heuristic = not hasattr(policy_or_fn, "rollout")
    h = None if is_nn_heuristic else policy_or_fn.encode(state)

    for _ in range(env.max_steps):
        if env.done.all():
            break
        if is_nn_heuristic:
            # .item() indispensable : env.dist[0, env.cur_node] avec un
            # tenseur d'indices renvoie (1, N) et non (N,), ce qui faisait
            # que `d_masked[0] = inf` écrasait toute la ligne au lieu du
            # seul dépôt (tout devenait invalide -> tournée vide).
            cur = int(env.cur_node.item())
            d = env.dist[0, cur].clone()
            d_masked = torch.where(state["mask"][0], d, torch.full_like(d, float("inf")))
            if state["mask"][0, 1:].any():
                d_masked[0] = float("inf")   # ne pas rentrer au dépôt si un client est servable
            action = d_masked.argmin().unsqueeze(0)
        else:
            logits = policy_or_fn.decode_step(h, state)
            action = logits.argmax(dim=-1) if greedy else \
                torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)

        route.append(int(action.item()))
        state, _ = env.step(action)

    return route, env.metrics()


def plot_route(instance, route, ax=None, title=None, save_path=None):
    """
    Trace la tournée. Chaque passage par le dépôt commence une nouvelle
    couleur : on visualise ainsi les tournées successives (recharges).
    """
    _apply_style()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 7))

    loc = instance.locations
    is_dyn = instance.is_dynamic

    # --- Segments, colorés par tournée ---
    tour_idx = 0
    for a, b in zip(route[:-1], route[1:]):
        if a == b:
            continue  # attente sur place : pas de déplacement
        color = TOUR_COLORS[tour_idx % len(TOUR_COLORS)]
        ax.annotate("", xy=loc[b], xytext=loc[a],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.6,
                                     alpha=0.85, shrinkA=6, shrinkB=6))
        if b == 0:
            tour_idx += 1

    static = loc[1:][~is_dyn[1:]]
    dynamic = loc[1:][is_dyn[1:]]
    if len(static):
        ax.scatter(static[:, 0], static[:, 1], s=70, color=TEAL,
                   edgecolor="white", linewidth=0.8, label="Client statique", zorder=4)
    if len(dynamic):
        ax.scatter(dynamic[:, 0], dynamic[:, 1], s=70, color=ACCENT,
                   edgecolor="white", linewidth=0.8, label="Client dynamique", zorder=4)
    ax.scatter(*loc[0], s=340, marker="*", color="#D64545",
               edgecolor=MIDNIGHT, linewidth=1.2, label="Dépôt", zorder=6)

    ax.set_title(title or "Tournée", fontsize=13, color=MIDNIGHT, fontweight="bold")
    leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.95,
                     facecolor="white", edgecolor="#DDDDDD")
    leg.set_zorder(10)
    ax.set_xticks([]); ax.set_yticks([])
    _clean_axes(ax, keep=())

    if created:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    return ax


def compare_routes(instance, policy, device="cpu", save_path=None):
    """Compare côte à côte la tournée du modèle RL et celle du plus proche voisin."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    route_nn, met_nn = rollout_single(lambda: None, instance, device)
    plot_route(instance, route_nn, ax=axes[0],
               title=f"Plus proche voisin — distance {met_nn['distance'].item():.0f}")

    route_rl, met_rl = rollout_single(policy, instance, device, greedy=True)
    plot_route(instance, route_rl, ax=axes[1],
               title=f"Modèle RL+GNN — distance {met_rl['distance'].item():.0f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    plt.show()
    plt.close(fig)
    return met_nn, met_rl


def plot_training_curves(history, nn_reference=None, save_path=None):
    """Courbes d'entraînement : distance, écart vs référence, perte."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ep = [h["epoch"] for h in history]

    ax = axes[0]
    ax.plot(ep, [h["eval_distance"] for h in history], color=DEEP, lw=2.2,
            label="Éval (glouton)")
    ax.plot(ep, [h["train_distance"] for h in history], color=TEAL, lw=1.4,
            alpha=0.65, label="Entraînement")
    if nn_reference is not None:
        ax.axhline(nn_reference, color=ACCENT, ls="--", lw=2, label="Plus proche voisin")
    ax.set_title("Distance totale", fontsize=13, color=MIDNIGHT, fontweight="bold")
    ax.set_xlabel("Époque"); ax.set_ylabel("Distance")
    ax.legend(fontsize=9, frameon=False)
    _clean_axes(ax)

    ax = axes[1]
    ax.plot(ep, [h["gap_vs_nn_pct"] for h in history], color=DEEP, lw=2.2)
    ax.axhline(0, color=ACCENT, ls="--", lw=2)
    ax.fill_between(ep, [h["gap_vs_nn_pct"] for h in history], 0,
                    color=DEEP, alpha=0.12)
    ax.set_title("Écart vs plus proche voisin", fontsize=13, color=MIDNIGHT, fontweight="bold")
    ax.set_xlabel("Époque"); ax.set_ylabel("Écart (%)")
    _clean_axes(ax)

    ax = axes[2]
    ax.plot(ep, [h["loss"] for h in history], color=GRAY, lw=1.6)
    upd = [h["epoch"] for h in history if h.get("baseline_updated")]
    for u in upd:
        ax.axvline(u, color=ACCENT, alpha=0.35, lw=1)
    ax.set_title(f"Perte (| = maj baseline, {len(upd)})",
                 fontsize=13, color=MIDNIGHT, fontweight="bold")
    ax.set_xlabel("Époque"); ax.set_ylabel("Perte")
    _clean_axes(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    plt.show()
    plt.close(fig)
