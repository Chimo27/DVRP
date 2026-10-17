"""
Étape 12 — Sensibilité au degré de dynamisme.

    python scripts/12_dod_sweep.py --checkpoint outputs/final/policy_principal.pt

Le degré de dynamisme (DoD) est le paramètre qui définit le problème traité :
c'est lui qui distingue le DVRP du CVRP statique. Le rapport le présente
comme central, mais aucune expérience ne le fait varier systématiquement.
Ce script comble cette lacune.

DEUX MODES
----------
  --mode eval   (défaut) : évalue une politique DÉJÀ entraînée sur des
                 instances de DoD croissant. Rapide (quelques minutes).
                 Mesure la robustesse au dynamisme, mais la politique a été
                 entraînée à un DoD unique : l'écart observé mélange donc
                 l'effet du dynamisme sur la difficulté du problème et
                 l'effet du décalage entraînement/test.

  --mode train  : entraîne une politique par valeur de DoD, chacune évaluée
                 dans son propre régime. Seul mode permettant de conclure
                 sur l'effet propre du dynamisme, au prix de K entraînements
                 complets (plusieurs heures).

Dans les deux cas, l'heuristique du plus proche voisin est évaluée sur les
mêmes instances : l'écart RL/heuristique reste donc interprétable à DoD
donné, même en mode `eval`.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.generator import generate_dataset
from rl.env import DVRPEnv
from rl.baselines import nearest_neighbor_policy
from rl.train import train, load_checkpoint

TEST_SEED_BASE = 40_000_000


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


@torch.no_grad()
def evaluate_at_dod(policy, dod, n_customers, n_instances, device,
                    congestion_mode, objective):
    """Évalue politique et heuristique sur des instances de degré `dod`."""
    insts = generate_dataset(
        n_instances, base_seed=TEST_SEED_BASE + int(dod * 1000),
        n_customers=n_customers, degree_of_dynamism=dod,
    )

    env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                  objective=objective, seed=0)
    _, met_nn = nearest_neighbor_policy(env)

    env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                  objective=objective, seed=0, information=policy.information)
    policy.eval()
    _, _, _, met_rl = policy.rollout(env, greedy=True)

    # Degré de dynamisme effectif réellement obtenu (pondéré par l'instant
    # de révélation), qui diffère du DoD nominal demandé.
    edd = float(np.mean([i.metadata["dod_stats"]["effective_dod"] for i in insts]))
    dod_reel = float(np.mean([i.metadata["dod_stats"]["dod_actual"] for i in insts]))

    rl_d = met_rl["distance"].mean().item()
    nn_d = met_nn["distance"].mean().item()
    rl_m = met_rl["makespan"].mean().item()
    nn_m = met_nn["makespan"].mean().item()

    return {
        "dod_nominal": dod,
        "dod_reel": dod_reel,
        "edd": edd,
        "rl_distance": rl_d,
        "nn_distance": nn_d,
        "gap_distance_pct": (rl_d - nn_d) / nn_d * 100,
        "rl_makespan": rl_m,
        "nn_makespan": nn_m,
        "gap_makespan_pct": (rl_m - nn_m) / nn_m * 100,
        "rl_unserved": met_rl["unserved"].mean().item(),
        "rl_wait": met_rl["wait"].mean().item(),
        "nn_wait": met_nn["wait"].mean().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval", "train"], default="eval")
    p.add_argument("--checkpoint", type=str,
                    default="outputs/final/policy_principal.pt")
    p.add_argument("--dods", type=float, nargs="+",
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--n-instances", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300,
                    help="mode train uniquement")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=20)
    p.add_argument("--congestion-mode", type=str, default="multiplicative")
    p.add_argument("--objective", type=str, default="time")
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="outputs/final/dod_sweep.json")
    args, _ = p.parse_known_args()

    t0 = time.time()
    results = {}

    print(f"Mode : {args.mode} | {args.n_customers} clients | device={args.device}")
    print(f"{'DoD':>6} {'DoD réel':>9} {'EDD':>7} {'RL dist':>9} {'PPV dist':>9} "
          f"{'écart':>8} {'écart mks':>10}")
    print("-" * 64)

    for dod in args.dods:
        if args.mode == "train":
            ckpt = f"outputs/final/dod_sweep/policy_dod{int(dod*100):03d}.pt"
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            policy, _ = train(
                n_customers=args.n_customers, epochs=args.epochs,
                batch_size=args.batch_size, batches_per_epoch=args.batches_per_epoch,
                device=args.device, seed=0,
                congestion_mode=args.congestion_mode, objective=args.objective,
                instance_overrides={"degree_of_dynamism": dod},
                checkpoint_path=ckpt, verbose=False,
            )
        else:
            policy, _ = load_checkpoint(args.checkpoint, device=args.device)

        r = evaluate_at_dod(policy, dod, args.n_customers, args.n_instances,
                            args.device, args.congestion_mode, args.objective)
        results[f"{dod:.1f}"] = r

        print(f"{dod:>6.1f} {r['dod_reel']:>9.3f} {r['edd']:>7.3f} "
              f"{r['rl_distance']:>9.1f} {r['nn_distance']:>9.1f} "
              f"{r['gap_distance_pct']:>+7.1f}% {r['gap_makespan_pct']:>+9.1f}%")

    payload = {
        "meta": {
            "mode": args.mode,
            "checkpoint": args.checkpoint if args.mode == "eval" else None,
            "n_customers": args.n_customers,
            "n_instances": args.n_instances,
            "congestion_mode": args.congestion_mode,
            "objective": args.objective,
            "duration_s": time.time() - t0,
            "note": ("mode=eval : la politique a ete entrainee a un DoD unique, "
                     "l'ecart observe melange effet du dynamisme et decalage "
                     "entrainement/test. mode=train : une politique par DoD, "
                     "chacune evaluee dans son propre regime."),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_to_py(payload), f, indent=2, ensure_ascii=False)
    print(f"\nExporté : {args.out}")


if __name__ == "__main__":
    main()
