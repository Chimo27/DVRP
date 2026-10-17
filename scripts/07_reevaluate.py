"""
Ré-évaluation des modèles DÉJÀ ENTRAÎNÉS, avec congestion cohérente.

    python scripts/07_reevaluate.py --device cuda

POURQUOI CE SCRIPT
------------------
Dans la campagne initiale (06_final_experiments.py), l'évaluation créait les
environnements de test SANS congestion, alors que le run « principal » avait
été entraîné AVEC congestion. Le modèle principal était donc testé dans des
conditions différentes de son entraînement, ce qui :

  - sous-estime probablement sa performance réelle en conditions congestionnées,
  - et surtout rend NON CONCLUANTE la comparaison avec l'ablation
    « sans_congestion » : l'écart observé pouvait venir du décalage
    entraînement/test plutôt que de la congestion elle-même.

Ce script ré-évalue chaque modèle DANS SES PROPRES CONDITIONS d'entraînement,
et évalue en plus le modèle principal dans les deux régimes pour mesurer sa
robustesse au changement de conditions.

Aucun ré-entraînement : on recharge les checkpoints existants (quelques
minutes seulement).
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from rl.train import load_checkpoint
from rl.evaluate import evaluate_all, evaluate_generalization, evaluate_domain_robustness

OUT_DIR = "outputs/final"

# Conditions d'entraînement de chaque run (cf. 06_final_experiments.py)
RUN_CONDITIONS = {
    "principal":        {"congestion_mode": "multiplicative", "objective": "time"},
    "sans_congestion":  {"congestion_mode": "off",            "objective": "time"},
    "domain_rand":      {"congestion_mode": "multiplicative", "objective": "time"},
}


def _to_py(o):
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-instances", type=int, default=256)
    args, _ = p.parse_known_args()

    out = {}

    for run_name, cond in RUN_CONDITIONS.items():
        ckpt = f"{OUT_DIR}/policy_{run_name}.pt"
        if not os.path.exists(ckpt):
            print(f"[!] Checkpoint absent, run ignoré : {ckpt}")
            continue

        policy, _ = load_checkpoint(ckpt, device=args.device)
        cm = cond["congestion_mode"]

        print("\n" + "=" * 68)
        print(f"RUN « {run_name} »  —  évalué dans ses conditions d'entraînement "
              f"(congestion={cm})")
        print("=" * 68)

        entry = {"train_conditions": cond}

        entry["test_matched"] = evaluate_all(
            policy, n_instances=args.n_instances, device=args.device,
            instance_overrides={"n_customers": 20},
            congestion_mode=cm, objective=cond["objective"], verbose=True,
        )

        print(f"\n[{run_name}] Généralisation (congestion={cm})...")
        entry["generalization_matched"] = evaluate_generalization(
            policy, sizes=(10, 20, 50), n_instances=128, device=args.device,
            congestion_mode=cm, verbose=True,
        )

        out[run_name] = entry

    # --- Transfert : le modèle principal face aux deux régimes ---
    ckpt = f"{OUT_DIR}/policy_principal.pt"
    if os.path.exists(ckpt):
        print("\n" + "=" * 68)
        print("TRANSFERT — modèle principal évalué dans les DEUX régimes")
        print("=" * 68)
        policy, _ = load_checkpoint(ckpt, device=args.device)
        transfer = {}
        for cm in ["multiplicative", "off"]:
            print(f"\n  congestion = {cm}")
            transfer[cm] = evaluate_all(
                policy, n_instances=args.n_instances, device=args.device,
                instance_overrides={"n_customers": 20},
                congestion_mode=cm, objective="time", verbose=True,
            )
        out["_transfer_principal"] = transfer

        print("\nRobustesse (congestion active)...")
        out["_robustness_principal_congested"] = evaluate_domain_robustness(
            policy, n_instances=128, device=args.device,
            congestion_mode="multiplicative", verbose=True,
        )

    path = f"{OUT_DIR}/reevaluation.json"
    with open(path, "w") as f:
        json.dump(_to_py(out), f, indent=2, ensure_ascii=False)
    print(f"\nExporté : {path}")


if __name__ == "__main__":
    main()
