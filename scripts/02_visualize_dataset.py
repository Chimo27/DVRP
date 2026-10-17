"""
Étape 2 — Visualisation du dataset et comparaison des distributions.

    python scripts/02_visualize_dataset.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")   # sauvegarde fichier ; retirer cette ligne en notebook

from data.generator import generate_instance
from viz.instance_viz import (plot_instance_dashboard, compare_spatial_methods,
                               print_stats_table)

OUT = "outputs/figures"

def main(n_customers=30):
    os.makedirs(OUT, exist_ok=True)

    inst = generate_instance(seed=42, n_customers=n_customers)
    print(inst.summary())
    plot_instance_dashboard(inst, save_path=f"{OUT}/instance_dashboard.png")

    print("\nComparaison des méthodes spatiales :\n")
    stats = compare_spatial_methods(n_customers=60, seed=0,
                                    save_path=f"{OUT}/spatial_comparison.png")
    print_stats_table(stats)

if __name__ == "__main__":
    main()
