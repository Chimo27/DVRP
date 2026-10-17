"""
Étape 10 — Ablation : baseline gloutonne figée vs mobile.

    python scripts/10_baseline_ablation.py --n-seeds 3 --epochs 300

Teste expérimentalement l'argument analytique de la section « Baseline
gloutonne figée » du rapport : la baseline mobile (recalculée à chaque batch
avec la politique courante) ferait s'éteindre le gradient à mesure que la
politique devient déterministe, contrairement à la baseline figée.

Nécessite le patch de rl/train.py ajoutant le paramètre `baseline_mode`
("frozen" | "moving"). Sans ce patch, seul le mode frozen est disponible.

Le script entraîne les deux variantes sur les mêmes graines (appariement),
avec le MÊME jeu de test à l'arrivée, et compare :
  - la courbe de progression (le mode moving devrait plafonner plus tôt) ;
  - le nombre de "mises à jour" de baseline qui auraient eu lieu au sens du
    critère figé (calculé même en mode moving, à titre de suivi, mais non
    utilisé dans la perte de ce mode) ;
  - la distance finale sur le jeu de test, avec test t apparié.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy import stats as scipy_stats

from rl.train import train
from rl.evaluate import evaluate_all

TEST_SEED_BASE = 20_000_000


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


def run_one(mode, seed, epochs, n_customers, batch_size, batches_per_epoch,
            device, n_test_instances, out_dir):
    ckpt = f"{out_dir}/policy_baseline_{mode}_seed{seed}.pt"
    print(f"\n--- baseline={mode}  graine={seed} ---")

    t0 = time.time()
    policy, history = train(
        n_customers=n_customers, epochs=epochs, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch, device=device, seed=seed,
        congestion_mode="multiplicative", objective="time",
        baseline_mode=mode, checkpoint_path=ckpt, verbose=False,
    )
    duration = time.time() - t0

    test_res = evaluate_all(
        policy, n_instances=n_test_instances, device=device,
        seed_base=TEST_SEED_BASE, instance_overrides={"n_customers": n_customers},
        congestion_mode="multiplicative", objective="time", verbose=False,
    )
    rl = test_res["rl_gnn_glouton"]

    # Progression : distance d'évaluation à intervalles réguliers, pour
    # visualiser si le mode "moving" plafonne plus tôt.
    stride = max(1, epochs // 20)
    progression = [{"epoch": h["epoch"], "eval_distance": h["eval_distance"]}
                    for h in history if h["epoch"] % stride == 0]

    n_updates = sum(1 for h in history if h.get("baseline_updated"))

    print(f"    distance test={rl['distance_moy']:.1f}  "
          f"maj_baseline_figee={n_updates}  ({duration/60:.1f} min)")

    return {
        "mode": mode, "seed": seed, "duration_s": duration,
        "test_distance": rl["distance_moy"],
        "test_makespan": rl["makespan_moy"],
        "n_baseline_updates_if_frozen": n_updates,
        "progression": progression,
        "checkpoint": ckpt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=20)
    p.add_argument("--n-test-instances", type=int, default=256)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="outputs/final/baseline_ablation")
    args, _ = p.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = {"frozen": [], "moving": []}
    t_campaign = time.time()

    for seed in range(args.n_seeds):
        for mode in ["frozen", "moving"]:
            r = run_one(mode, seed, args.epochs, args.n_customers, args.batch_size,
                        args.batches_per_epoch, args.device, args.n_test_instances,
                        args.out_dir)
            runs[mode].append(r)

    d_frozen = [r["test_distance"] for r in runs["frozen"]]
    d_moving = [r["test_distance"] for r in runs["moving"]]

    print(f"\nfrozen : {np.mean(d_frozen):.1f} +/- {np.std(d_frozen):.1f}")
    print(f"moving : {np.mean(d_moving):.1f} +/- {np.std(d_moving):.1f}")

    test_result = None
    if len(d_frozen) == len(d_moving) and len(d_frozen) >= 2:
        t_stat, p_val = scipy_stats.ttest_rel(d_moving, d_frozen)
        test_result = {
            "t_statistic": float(t_stat), "p_value": float(p_val),
            "significant_at_0.05": bool(p_val < 0.05),
            "mean_diff_moving_minus_frozen": float(np.mean(d_moving) - np.mean(d_frozen)),
        }
        sig = "significatif" if p_val < 0.05 else "non significatif"
        print(f"\ntest t apparié (moving - frozen) : "
              f"diff = {test_result['mean_diff_moving_minus_frozen']:+.1f}, "
              f"p = {p_val:.4f} ({sig})")

    payload = {
        "meta": {
            "n_seeds": args.n_seeds, "epochs": args.epochs,
            "n_customers": args.n_customers,
            "duration_total_s": time.time() - t_campaign,
        },
        "runs": runs,
        "test": test_result,
    }

    out_path = f"{args.out_dir}/resultats_baseline_ablation.json"
    with open(out_path, "w") as f:
        json.dump(_to_py(payload), f, indent=2, ensure_ascii=False)
    print(f"\nExporté : {out_path}")


if __name__ == "__main__":
    main()
