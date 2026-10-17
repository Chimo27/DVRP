"""
Étape 5 — Comparer la solution du modèle entraîné à l'optimum (ou
quasi-optimum) sur une instance dérivée de Solomon.

    python scripts/05_compare_solomon.py --solomon path/vers/C101.txt --n 12

RAPPEL : les fenêtres de temps du fichier Solomon sont ignorées (notre
modèle ne les gère pas). On compare donc à l'optimum de la version SANS
fenêtres, recalculé ici avec OR-Tools — PAS au chiffre publié de Solomon
(qui est l'optimum AVEC fenêtres, donc plus élevé, pas comparable).
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.train import load_checkpoint
from rl.infer import solve_instance, print_solution
from rl.exact_solver import solve_gls as solve_exact  # [révision] GLS : pas de preuve d optimalite
from data.solomon_loader import load_solomon_instance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="outputs/checkpoints/policy_20.pt")
    p.add_argument("--solomon", type=str, required=True, help="Chemin du fichier Solomon (.txt)")
    p.add_argument("--n", type=int, default=12,
                    help="Nb de clients à garder (<=12-15 pour un optimum PROUVÉ par OR-Tools)")
    p.add_argument("--time-limit", type=int, default=30, help="Limite de temps OR-Tools (s)")
    p.add_argument("--device", type=str, default="cpu")
    args, _ = p.parse_known_args()

    # --- 1. Charger l'instance (fenêtres de temps ignorées, cf. solomon_loader.py) ---
    instance = load_solomon_instance(args.solomon, n_customers=args.n)
    print(instance.summary())
    print(f"({instance.metadata['note']})\n")

    # --- 2. Charger le modèle entraîné, pure inférence ---
    policy, _ = load_checkpoint(args.checkpoint, device=args.device)
    rl_tours, rl_distance = solve_instance(policy, instance, device=args.device)

    # --- 3. Solveur de référence (optimum si n petit, sinon best-effort) ---
    ref_distance, ref_tours, proven_optimal = solve_exact(instance, time_limit_s=args.time_limit)

    # --- 4. Comparaison ---
    print("=" * 60)
    print("SOLUTION DU MODÈLE RL+GNN")
    print("=" * 60)
    print_solution(rl_tours, instance)

    print("\n" + "=" * 60)
    label = "SOLUTION OPTIMALE (prouvée)" if proven_optimal else \
            "SOLUTION DE RÉFÉRENCE (OR-Tools, PAS garantie optimale — augmenter --time-limit ou réduire --n)"
    print(label)
    print("=" * 60)
    print_solution(ref_tours, instance)

    gap = (rl_distance - ref_distance) / ref_distance * 100
    print("\n" + "=" * 60)
    print(f"Distance modèle RL+GNN      : {rl_distance:.2f}")
    print(f"Distance référence          : {ref_distance:.2f}"
          + ("  (optimum prouvé)" if proven_optimal else "  (non prouvée optimale)"))
    print(f"Écart du modèle             : {gap:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
