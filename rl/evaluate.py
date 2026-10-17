"""
Évaluation du modèle entraîné face aux heuristiques de référence.

Toutes les méthodes sont évaluées sur EXACTEMENT le même jeu d'instances
(mêmes seeds), donc les écarts sont directement interprétables.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.generator import generate_dataset
from rl.env import DVRPEnv
from rl.baselines import BASELINES, nearest_neighbor_policy

# Seeds réservées à l'évaluation, disjointes de celles de l'entraînement.
TEST_SEED_BASE = 20_000_000


@torch.no_grad()
def evaluate_all(policy, n_instances=256, device="cpu", seed_base=TEST_SEED_BASE,
                  instance_overrides=None, verbose=True,
                  congestion_mode=None, objective=None):
    overrides = dict(instance_overrides or {})
    insts = generate_dataset(n_instances, base_seed=seed_base, **overrides)

    results = {}

    for name, fn in BASELINES.items():
        env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                       objective=objective, seed=seed_base)
        reward, met = fn(env)
        results[name] = _summarize(reward, met)

    env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                   objective=objective, seed=seed_base, information=policy.information)
    policy.eval()
    _, reward, _, met = policy.rollout(env, greedy=True)
    results["rl_gnn_glouton"] = _summarize(reward, met)

    if verbose:
        print_comparison(results)
    return results


def _summarize(reward, met):
    return {
        "distance_moy": met["distance"].mean().item(),
        "distance_ecart_type": met["distance"].std().item(),
        "makespan_moy": met["makespan"].mean().item(),
        "non_servis_moy": met["unserved"].mean().item(),
        "taux_completion": met["completed"].mean().item() * 100,
        "reward_moy": reward.mean().item(),
    }


def print_comparison(results, reference="nearest_neighbor"):
    methods = list(results.keys())
    metrics = ["distance_moy", "makespan_moy", "non_servis_moy", "taux_completion"]

    w = 20
    header = f"{'Méthode':<22}" + "".join(f"{m:>{w}}" for m in metrics) + f"{'vs réf.':>12}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    ref_dist = results.get(reference, {}).get("distance_moy")
    for m in methods:
        row = f"{m:<22}"
        for met in metrics:
            row += f"{results[m][met]:>{w}.2f}"
        if ref_dist:
            gap = (results[m]["distance_moy"] - ref_dist) / ref_dist * 100
            row += f"{gap:>+11.1f}%"
        print(row)
    print("=" * len(header))
    print(f"(« vs réf. » : écart de distance vs '{reference}' — négatif = meilleur)")


@torch.no_grad()
def evaluate_generalization(policy, sizes=(10, 20, 50), n_instances=128,
                             device="cpu", verbose=True, congestion_mode=None):
    """
    Teste le modèle sur des tailles d'instances différentes de celle
    d'entraînement — un GNN devrait généraliser, c'est un de ses arguments
    principaux face à une architecture à taille fixe.
    """
    out = {}
    for n in sizes:
        res = evaluate_all(policy, n_instances=n_instances, device=device,
                            instance_overrides={"n_customers": n}, verbose=False,
                            congestion_mode=congestion_mode)
        nn_d = res["nearest_neighbor"]["distance_moy"]
        rl_d = res["rl_gnn_glouton"]["distance_moy"]
        out[n] = {
            "rl_distance": rl_d,
            "nn_distance": nn_d,
            "gap_pct": (rl_d - nn_d) / nn_d * 100,
        }
        if verbose:
            print(f"{n:4d} clients | RL={rl_d:8.1f} | NN={nn_d:8.1f} | "
                  f"écart={out[n]['gap_pct']:+6.1f}%")
    return out


# ---------------------------------------------------------------------------
# Configurations de domaine "canoniques" utilisées pour mesurer la
# robustesse — délibérément DIFFÉRENTES de DOMAIN_RANGES (rl/train.py
# n'y a jamais entraîné exactement ces combinaisons), pour que le résultat
# ne soit pas un simple rappel de configurations déjà vues.
ROBUSTNESS_CONFIGS = {
    "petite_uniforme":    {"n_customers": 10, "spatial_method": "uniform",   "degree_of_dynamism": 0.2},
    "petite_clusters":    {"n_customers": 10, "spatial_method": "clustered", "degree_of_dynamism": 0.2},
    "moyenne_uniforme":   {"n_customers": 25, "spatial_method": "uniform",   "degree_of_dynamism": 0.5},
    "moyenne_clusters":   {"n_customers": 25, "spatial_method": "clustered", "degree_of_dynamism": 0.5},
    "grande_uniforme":    {"n_customers": 50, "spatial_method": "uniform",   "degree_of_dynamism": 0.5},
    "grande_dynamisme_fort": {"n_customers": 25, "spatial_method": "uniform", "degree_of_dynamism": 0.8},
}


@torch.no_grad()
def evaluate_domain_robustness(policy, configs=None, n_instances=128,
                                device="cpu", seed_base=30_000_000, verbose=True,
                                congestion_mode=None):
    """
    Évalue le modèle sur plusieurs configurations de domaine distinctes
    (taille, distribution spatiale, degré de dynamisme), toujours comparé
    au plus proche voisin DANS LA MÊME configuration.

    Sert à quantifier l'effet de la randomisation de domaine : un modèle
    entraîné sans elle a typiquement un écart qui se dégrade fortement hors
    de sa configuration d'entraînement ; un modèle entraîné avec devrait
    rester proche (voire meilleur) sur l'ensemble des configurations.
    """
    configs = configs or ROBUSTNESS_CONFIGS
    results = {}

    for name, overrides in configs.items():
        insts = generate_dataset(n_instances, base_seed=seed_base, **overrides)

        env_nn = DVRPEnv(insts, device=device, congestion_mode=congestion_mode, seed=seed_base)
        _, met_nn = nearest_neighbor_policy(env_nn)
        nn_d = met_nn["distance"].mean().item()

        env_rl = DVRPEnv(insts, device=device, congestion_mode=congestion_mode, seed=seed_base,
                         information=policy.information)
        policy.eval()
        _, reward, _, met_rl = policy.rollout(env_rl, greedy=True)
        rl_d = met_rl["distance"].mean().item()
        unserved = met_rl["unserved"].mean().item()

        results[name] = {
            "config": overrides,
            "rl_distance": rl_d,
            "nn_distance": nn_d,
            "gap_pct": (rl_d - nn_d) / nn_d * 100,
            "unserved": unserved,
        }

    if verbose:
        w = 22
        header = f"{'Configuration':<24}{'RL':>{w}}{'Plus proche voisin':>{w}}{'Écart':>12}"
        print(header)
        print("-" * len(header))
        for name, r in results.items():
            print(f"{name:<24}{r['rl_distance']:>{w}.1f}{r['nn_distance']:>{w}.1f}"
                  f"{r['gap_pct']:>+11.1f}%")
        gaps = [r["gap_pct"] for r in results.values()]
        print("-" * len(header))
        print(f"Écart moyen : {sum(gaps)/len(gaps):+.1f}%  |  "
              f"écart-type : {(sum((g-sum(gaps)/len(gaps))**2 for g in gaps)/len(gaps))**0.5:.1f} "
              "(faible écart-type = robustesse across configurations)")

    return results
