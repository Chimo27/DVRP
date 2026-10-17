"""
Visualisation des instances DVRP et des datasets.

Palette et style centralisés ici pour que toutes les figures du projet
(rapport, présentation) soient cohérentes.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Palette du projet ---
DEEP = "#065A82"
TEAL = "#1C7293"
MIDNIGHT = "#21295C"
ACCENT = "#F2A65A"
GRAY = "#8A97A0"
LIGHT = "#E8F1F5"

STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#CCCCCC",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _apply_style():
    plt.rcParams.update(STYLE)


def _clean_axes(ax, keep=("bottom", "left")):
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


# ===========================================================================
def plot_instance(instance, ax=None, title=None, show_ids=False):
    """Carte d'une instance : dépôt, clients statiques et dynamiques."""
    _apply_style()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6, 6))

    loc = instance.locations
    is_dyn = instance.is_dynamic
    depot = loc[0]

    static = loc[1:][~is_dyn[1:]]
    dynamic = loc[1:][is_dyn[1:]]

    if len(static):
        ax.scatter(static[:, 0], static[:, 1], s=70, color=TEAL,
                   edgecolor="white", linewidth=0.8, label="Client statique", zorder=3)
    if len(dynamic):
        ax.scatter(dynamic[:, 0], dynamic[:, 1], s=70, color=ACCENT,
                   edgecolor="white", linewidth=0.8, label="Client dynamique", zorder=3)
    ax.scatter(*depot, s=320, marker="*", color="#D64545",
               edgecolor=MIDNIGHT, linewidth=1.2, label="Dépôt", zorder=5)

    if show_ids:
        for i in range(1, len(loc)):
            ax.annotate(str(i), loc[i], fontsize=7, color=MIDNIGHT,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_title(title or f"Instance — {instance.n_customers()} clients",
                 fontsize=13, color=MIDNIGHT, fontweight="bold")
    # Fond opaque : sans cadre, les marqueurs de la légende se confondent
    # avec de vrais points de données (une étoile de légende ressemblait à
    # un second dépôt).
    leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.95,
                     facecolor="white", edgecolor="#DDDDDD")
    leg.set_zorder(10)
    ax.set_xticks([]); ax.set_yticks([])
    _clean_axes(ax, keep=())

    if created:
        plt.tight_layout()
    return ax


def plot_instance_dashboard(instance, save_path=None):
    """
    Tableau de bord d'une instance : carte, demandes, arrivées dynamiques,
    distances au dépôt.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    plot_instance(instance, ax=axes[0, 0])

    # --- Demandes ---
    ax = axes[0, 1]
    d = instance.demands[1:]
    bins = np.arange(d.min(), d.max() + 2) - 0.5
    ax.hist(d, bins=bins, color=TEAL, edgecolor="white", rwidth=0.85)
    ax.set_title("Distribution des demandes", fontsize=13, color=MIDNIGHT, fontweight="bold")
    ax.set_xlabel("Demande"); ax.set_ylabel("Nb clients")
    _clean_axes(ax)

    # --- Arrivées dynamiques ---
    ax = axes[1, 0]
    dyn_t = instance.appearance_times[1:][instance.is_dynamic[1:]]
    if len(dyn_t):
        ax.hist(dyn_t, bins=12, color=ACCENT, edgecolor="white")
        s = instance.metadata.get("dod_stats", {})
        ax.set_title(f"Arrivées dynamiques (DoD={s.get('dod_actual','?')}, "
                     f"EDD={s.get('effective_dod','?')})",
                     fontsize=13, color=MIDNIGHT, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Instance statique", ha="center", va="center",
                transform=ax.transAxes, color=GRAY)
        ax.set_title("Arrivées dynamiques", fontsize=13, color=MIDNIGHT, fontweight="bold")
    ax.set_xlabel("Instant d'apparition (min)"); ax.set_ylabel("Nb clients")
    _clean_axes(ax)

    # --- Distances au dépôt ---
    ax = axes[1, 1]
    dd = instance.dist_matrix[0, 1:]
    ax.hist(dd, bins=12, color=DEEP, edgecolor="white")
    ax.axvline(dd.mean(), color=ACCENT, linestyle="--", linewidth=2,
               label=f"moyenne={dd.mean():.1f}")
    ax.set_title("Distances au dépôt", fontsize=13, color=MIDNIGHT, fontweight="bold")
    ax.set_xlabel("Distance"); ax.set_ylabel("Nb clients")
    ax.legend(fontsize=9, frameon=False)
    _clean_axes(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    plt.show()
    plt.close(fig)


def compare_spatial_methods(n_customers=60, seed=0, methods=("uniform", "clustered"),
                             save_path=None, **overrides):
    """Compare les méthodes spatiales à seed identique + statistiques."""
    from data.generator import generate_instance
    _apply_style()

    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5.5))
    if len(methods) == 1:
        axes = [axes]

    stats = {}
    for ax, m in zip(axes, methods):
        inst = generate_instance(seed=seed, n_customers=n_customers,
                                  spatial_method=m, **overrides)
        plot_instance(inst, ax=ax, title=m)

        cust = inst.locations[1:]
        pw = np.linalg.norm(cust[:, None, :] - cust[None, :, :], axis=-1)
        np.fill_diagonal(pw, np.inf)
        stats[m] = {
            "dist_moy_au_depot": float(inst.dist_matrix[0, 1:].mean()),
            "dist_moy_inter_clients": float(pw[np.isfinite(pw)].mean()),
            "plus_proche_voisin_moy": float(pw.min(axis=1).mean()),
            "dispersion_x": float(cust[:, 0].std()),
            "dispersion_y": float(cust[:, 1].std()),
        }

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {save_path}")
    plt.show()
    plt.close(fig)
    return stats


def print_stats_table(stats, col_width=22):
    """Affiche un dict {méthode: {métrique: valeur}} en tableau."""
    methods = list(stats.keys())
    metrics = list(next(iter(stats.values())).keys())
    header = f"{'Métrique':<26}" + "".join(f"{m:>{col_width}}" for m in methods)
    print(header)
    print("-" * len(header))
    for met in metrics:
        row = f"{met:<26}" + "".join(f"{stats[m][met]:>{col_width}.2f}" for m in methods)
        print(row)
