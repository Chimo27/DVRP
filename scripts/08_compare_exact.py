"""
Étape 8 — Comparaison au solveur exact (OR-Tools) sur petites instances.

    python scripts/08_compare_exact.py --checkpoint outputs/final/policy_principal.pt

C'est le complément identifié dans le rapport comme le plus utile à apporter
au protocole : situer l'écart de la politique apprise non plus seulement par
rapport à une heuristique gloutonne, mais par rapport à une référence de
qualité connue.

============================================================================
PRÉCISION INDISPENSABLE AVANT DE LIRE LES RÉSULTATS
============================================================================
Le solveur OR-Tools intégré (`rl/exact_solver.py`) résout un CVRP STATIQUE :
il connaît la position et la demande de tous les clients dès le départ, et ne
modélise ni l'instant de révélation r_i, ni la congestion dépendante de
l'heure. Il résout donc une version RELÂCHÉE du problème réellement traité
par la politique RL.

Conséquence directe : la distance retournée par OR-Tools est une BORNE
INFÉRIEURE de l'optimum du DVRP dynamique, pas l'optimum lui-même. Un vrai
solveur du DVRP dynamique devrait interdire de visiter un client avant r_i,
ce qu'OR-Tools ne fait pas ici. L'écart mesuré entre la politique et cette
borne est donc *plus grand* que l'écart réel à l'optimum dynamique — c'est
un majorant honnête de la sous-optimalité, pas une mesure exacte.

Ce script rapporte ce résultat sans le déguiser en "écart à l'optimum du
DVRP" : les tableaux et le JSON exporté utilisent systématiquement le terme
"borne inférieure" (lower bound).
============================================================================

Tailles couvertes : 8, 10, 12 clients (au-delà, `proven_optimal` devient
peu fiable — cf. rl/exact_solver.py). Le champ `proven_optimal` du JSON
exporté indique, instance par instance, si l'optimalité de cette borne a
réellement été prouvée ou seulement approchée dans le temps imparti.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from rl.train import load_checkpoint
from rl.exact_solver import solve_exact
from rl.baselines import nearest_neighbor_policy
from rl.env import DVRPEnv
from data.generator import generate_dataset

TEST_SEED_BASE = 20_000_000  # même plage que le jeu de test principal


def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return o


def rl_distance_single(policy, instance, device, congestion_mode, objective):
    """Distance parcourue par la politique (décodage glouton) sur UNE instance,
    dans le MÊME régime (congestion, objectif) que son entraînement."""
    env = DVRPEnv([instance], device=device, congestion_mode=congestion_mode,
                  objective=objective, seed=0, information=policy.information)
    policy.eval()
    with torch.no_grad():
        _, _, _, metrics = policy.rollout(env, greedy=True)
    return metrics["distance"].item()


def nn_distance_single(instance, device, congestion_mode, objective):
    env = DVRPEnv([instance], device=device, congestion_mode=congestion_mode,
                  objective=objective, seed=0)
    _, metrics = nearest_neighbor_policy(env)
    return metrics["distance"].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                    default="outputs/final/policy_principal.pt")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12])
    p.add_argument("--n-instances", type=int, default=30,
                    help="Nombre d'instances par taille")
    p.add_argument("--time-limit", type=int, default=30,
                    help="Budget OR-Tools par instance (secondes)")
    p.add_argument("--congestion-mode", type=str, default="multiplicative")
    p.add_argument("--objective", type=str, default="time")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str,
                    default="outputs/final/comparaison_exacte.json")
    args, _ = p.parse_known_args()

    policy, _ = load_checkpoint(args.checkpoint, device=args.device)

    results = {}
    t_start = time.time()

    for n in args.sizes:
        print(f"\n=== {n} clients ({args.n_instances} instances, "
              f"budget OR-Tools {args.time_limit}s/instance) ===")
        insts = generate_dataset(args.n_instances, base_seed=TEST_SEED_BASE + n,
                                  n_customers=n)

        rows = []
        for i, inst in enumerate(insts):
            t0 = time.time()
            lb_dist, _, proven = solve_exact(inst, time_limit_s=args.time_limit)
            rl_dist = rl_distance_single(policy, inst, args.device,
                                         args.congestion_mode, args.objective)
            nn_dist = nn_distance_single(inst, args.device,
                                         args.congestion_mode, args.objective)

            rows.append({
                "instance": i,
                "lower_bound": lb_dist,
                "proven_optimal": bool(proven),
                "rl_distance": rl_dist,
                "nn_distance": nn_dist,
                "gap_rl_vs_lb_pct": (rl_dist - lb_dist) / lb_dist * 100,
                "gap_nn_vs_lb_pct": (nn_dist - lb_dist) / lb_dist * 100,
            })
            dt = time.time() - t0
            print(f"  instance {i+1}/{len(insts)} : LB={lb_dist:.1f} "
                  f"({'prouvé' if proven else 'non prouvé'})  "
                  f"RL={rl_dist:.1f} (+{rows[-1]['gap_rl_vs_lb_pct']:.1f}%)  "
                  f"NN={nn_dist:.1f} (+{rows[-1]['gap_nn_vs_lb_pct']:.1f}%)  "
                  f"[{dt:.1f}s]")

        gaps_rl = [r["gap_rl_vs_lb_pct"] for r in rows]
        gaps_nn = [r["gap_nn_vs_lb_pct"] for r in rows]
        n_proven = sum(r["proven_optimal"] for r in rows)

        summary = {
            "n_customers": n,
            "n_instances": len(rows),
            "n_proven_optimal": n_proven,
            "proven_rate": n_proven / len(rows),
            "gap_rl_vs_lb_mean": float(np.mean(gaps_rl)),
            "gap_rl_vs_lb_std": float(np.std(gaps_rl)),
            "gap_nn_vs_lb_mean": float(np.mean(gaps_nn)),
            "gap_nn_vs_lb_std": float(np.std(gaps_nn)),
            "rows": rows,
        }
        results[str(n)] = summary

        print(f"  -> écart RL/borne : {summary['gap_rl_vs_lb_mean']:.1f}% "
              f"(+/- {summary['gap_rl_vs_lb_std']:.1f})  "
              f"| écart NN/borne : {summary['gap_nn_vs_lb_mean']:.1f}%  "
              f"| optimalité prouvée : {n_proven}/{len(rows)}")

    payload = {
        "meta": {
            "checkpoint": args.checkpoint,
            "congestion_mode": args.congestion_mode,
            "objective": args.objective,
            "time_limit_s": args.time_limit,
            "duration_s": time.time() - t_start,
            "warning": ("OR-Tools resout un CVRP statique (sans contrainte de "
                        "revelation r_i, sans congestion) : les valeurs "
                        "'lower_bound' sont une borne inferieure de l'optimum "
                        "du DVRP dynamique, pas l'optimum lui-meme."),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_to_py(payload), f, indent=2, ensure_ascii=False)
    print(f"\nExporté : {args.out}")


if __name__ == "__main__":
    main()
