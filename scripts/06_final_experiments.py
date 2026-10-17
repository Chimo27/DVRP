"""
Campagne expérimentale finale pour le rapport de stage.

Lance les entraînements « officiels » et exporte TOUTES les données
nécessaires à la rédaction du rapport dans un unique fichier JSON.

    python scripts/06_final_experiments.py --device cuda

Configuration retenue (décidée avec l'encadrante) :
    congestion : multiplicative
    objectif   : time  (cf. ci-dessous)

POURQUOI objectif="time" AVEC LA CONGESTION
--------------------------------------------
La congestion allonge le TEMPS de trajet, pas la distance. Avec un objectif
en distance pure, le modèle subirait les heures de pointe sans aucune raison
de les éviter : la congestion serait simulée mais invisible dans
l'apprentissage. En optimisant le makespan, le modèle a une incitation
réelle à composer avec les heures de pointe.

TROIS RUNS (le 1 est le résultat central, les 2 et 3 sont des ablations qui
renforcent l'analyse du rapport) :
    A. principal : congestion multiplicative + objectif temps
    B. ablation  : sans congestion (mesure l'effet de la congestion)
    C. ablation  : avec randomisation de domaine (mesure l'effet sur la
                    robustesse)
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from rl.train import train, load_checkpoint
from rl.evaluate import evaluate_all, evaluate_generalization, evaluate_domain_robustness
from data.generator import generate_dataset, generate_instance
from rl.env import DVRPEnv
from rl.baselines import nearest_neighbor_policy

OUT_DIR = "outputs/final"


def _to_py(obj):
    """Rend un objet sérialisable en JSON (numpy/torch -> python)."""
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def run_experiment(name, device, epochs, batch_size, batches_per_epoch,
                    n_customers, congestion_mode, objective,
                    domain_randomization, seed=0):
    print("\n" + "=" * 70)
    print(f"RUN « {name} »  |  congestion={congestion_mode}  objectif={objective}  "
          f"domain_rand={domain_randomization}")
    print("=" * 70)

    ckpt = f"{OUT_DIR}/policy_{name}.pt"
    t0 = time.time()

    policy, history = train(
        n_customers=n_customers, epochs=epochs, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch, device=device, seed=seed,
        congestion_mode=congestion_mode, objective=objective,
        domain_randomization=domain_randomization,
        checkpoint_path=ckpt, checkpoint_every=20, log_every=5,
    )
    duration = time.time() - t0

    # --- Évaluation sur jeu de test (seeds jamais vues) ---
    print(f"\n[{name}] Évaluation sur 256 instances de test...")
    test_results = evaluate_all(
        policy, n_instances=256, device=device,
        instance_overrides={"n_customers": n_customers}, verbose=True,
    )

    print(f"\n[{name}] Généralisation à d'autres tailles...")
    generalization = evaluate_generalization(
        policy, sizes=(10, 20, 50), n_instances=128, device=device, verbose=True,
    )

    return {
        "name": name,
        "config": {
            "n_customers": n_customers, "epochs": epochs,
            "batch_size": batch_size, "batches_per_epoch": batches_per_epoch,
            "congestion_mode": congestion_mode, "objective": objective,
            "domain_randomization": domain_randomization, "seed": seed,
            "device": device,
        },
        "duration_s": duration,
        "history": history,
        "test_results": test_results,
        "generalization": generalization,
        "checkpoint": ckpt,
    }


def sample_instance_table(n_customers=10, seed=42):
    """Extrait d'instance à reproduire dans le rapport (tableau)."""
    inst = generate_instance(seed=seed, n_customers=n_customers)
    rows = []
    for i in range(inst.n_total()):
        rows.append({
            "id": i,
            "type": "depot" if i == 0 else ("dynamique" if inst.is_dynamic[i] else "statique"),
            "x": round(float(inst.locations[i, 0]), 1),
            "y": round(float(inst.locations[i, 1]), 1),
            "demande": int(inst.demands[i]),
            "appearance_time": int(inst.appearance_times[i]),
            "service": int(inst.service_times[i]),
        })
    return {
        "seed": seed, "n_customers": n_customers,
        "capacity": inst.capacity, "total_demand": inst.total_demand(),
        "min_vehicles": inst.min_vehicles(),
        "horizon": list(inst.metadata["planning_horizon"]),
        "rows": rows,
    }


def congestion_effect_table(device="cpu"):
    """Effet mesuré de la congestion sur l'heuristique de référence."""
    insts = generate_dataset(256, base_seed=20_000_000, n_customers=20)
    out = {}
    for mode in ["off", "faithful", "normalized", "multiplicative"]:
        env = DVRPEnv(insts, device=device, congestion_mode=mode, seed=1)
        _, m = nearest_neighbor_policy(env)
        out[mode] = {
            "distance": float(m["distance"].mean()),
            "makespan": float(m["makespan"].mean()),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=20)
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ablations", action="store_true",
                    help="Ne lance que le run principal (plus rapide)")
    args, _ = p.parse_known_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device : {args.device}")
    if args.device == "cpu":
        print("ATTENTION : entraînement sur CPU, prévoir plusieurs heures. "
              "Activer le GPU si possible.")

    common = dict(
        device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch, n_customers=args.n_customers,
        seed=args.seed,
    )

    runs = {}

    # --- A. Run principal ---
    runs["principal"] = run_experiment(
        "principal", congestion_mode="multiplicative", objective="time",
        domain_randomization=False, **common,
    )

    if not args.skip_ablations:
        # --- B. Ablation : sans congestion ---
        runs["sans_congestion"] = run_experiment(
            "sans_congestion", congestion_mode="off", objective="time",
            domain_randomization=False, **common,
        )
        # --- C. Ablation : avec randomisation de domaine ---
        runs["domain_rand"] = run_experiment(
            "domain_rand", congestion_mode="multiplicative", objective="time",
            domain_randomization=True, **common,
        )

    # --- Données complémentaires pour le rapport ---
    print("\nCollecte des données complémentaires...")
    extras = {
        "instance_example": sample_instance_table(),
        "congestion_effect": congestion_effect_table(device=args.device),
    }

    # --- Robustesse du run principal ---
    print("\nRobustesse du modèle principal à travers les configurations...")
    policy, _ = load_checkpoint(runs["principal"]["checkpoint"], device=args.device)
    extras["robustness"] = evaluate_domain_robustness(
        policy, n_instances=128, device=args.device, verbose=True,
    )

    payload = _to_py({"runs": runs, "extras": extras})
    out_path = f"{OUT_DIR}/resultats_rapport.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"TERMINÉ. Résultats exportés : {out_path}")
    print("Checkpoints :", ", ".join(r["checkpoint"] for r in runs.values()))
    print("=" * 70)


if __name__ == "__main__":
    main()
