"""
Étape 9 — Répétition multi-graines et test statistique.

    python scripts/09_multi_seed.py --configs principal sans_congestion --n-seeds 5

Répond à la limite « chaque configuration correspond à un unique
entraînement » signalée dans le rapport. Chaque configuration listée est
entraînée K fois (une graine différente par répétition), puis évaluée sur
LE MÊME jeu de test à chaque fois, dans SON PROPRE régime de congestion
(cf. la limite « décalage entraînement/test » également signalée).

Les K graines sont partagées entre les configurations comparées (graine 0
pour la 1re répétition de chacune, graine 1 pour la 2e, etc.), ce qui permet
un test t APPARIÉ entre deux configurations : l'appariement neutralise la
variance liée à l'initialisation commune, et isole l'effet de la
configuration elle-même.

ATTENTION AU TEMPS DE CALCUL : K répétitions x 300 époques x plusieurs
configurations peut représenter plusieurs heures sur GPU. Le paramètre
--epochs permet de réduire la durée par entraînement si nécessaire ; dans ce
cas, le comparer explicitement à la campagne de 300 époques du rapport plutôt
que de le présenter comme équivalent.
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

# Configurations disponibles (cf. scripts/06_final_experiments.py)
CONFIGS = {
    "principal":       {"congestion_mode": "multiplicative", "objective": "time",
                         "domain_randomization": False},
    "sans_congestion": {"congestion_mode": "off",            "objective": "time",
                         "domain_randomization": False},
    "domain_rand":      {"congestion_mode": "multiplicative", "objective": "time",
                         "domain_randomization": True},
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
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return o


def mean_ci95(values):
    """Moyenne et demi-largeur de l'intervalle de confiance à 95% (loi t)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, float("nan")
    sem = values.std(ddof=1) / np.sqrt(n)
    half_width = sem * scipy_stats.t.ppf(0.975, df=n - 1)
    return float(mean), float(half_width)


def run_one(config_name, seed, epochs, n_customers, batch_size,
            batches_per_epoch, device, n_test_instances, out_dir):
    cfg = CONFIGS[config_name]
    ckpt = f"{out_dir}/policy_{config_name}_seed{seed}.pt"

    print(f"\n--- {config_name}  graine={seed} ---")
    t0 = time.time()
    policy, history = train(
        n_customers=n_customers, epochs=epochs, batch_size=batch_size,
        batches_per_epoch=batches_per_epoch, device=device, seed=seed,
        congestion_mode=cfg["congestion_mode"], objective=cfg["objective"],
        domain_randomization=cfg["domain_randomization"],
        checkpoint_path=ckpt, verbose=False,
    )
    duration = time.time() - t0

    test_res = evaluate_all(
        policy, n_instances=n_test_instances, device=device,
        seed_base=TEST_SEED_BASE, instance_overrides={"n_customers": n_customers},
        congestion_mode=cfg["congestion_mode"], objective=cfg["objective"],
        verbose=False,
    )

    rl = test_res["rl_gnn_glouton"]
    nn = test_res["nearest_neighbor"]
    print(f"    distance={rl['distance_moy']:.1f}  makespan={rl['makespan_moy']:.1f}  "
          f"({duration/60:.1f} min)")

    return {
        "seed": seed, "duration_s": duration,
        "final_train_distance": history[-1]["train_distance"],
        "test_distance": rl["distance_moy"],
        "test_makespan": rl["makespan_moy"],
        "nn_distance": nn["distance_moy"],
        "nn_makespan": nn["makespan_moy"],
        "gap_distance_pct": (rl["distance_moy"] - nn["distance_moy"])
                              / nn["distance_moy"] * 100,
        "checkpoint": ckpt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["principal", "sans_congestion"],
                    choices=list(CONFIGS))
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=20)
    p.add_argument("--n-test-instances", type=int, default=256)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="outputs/final/multi_seed")
    args, _ = p.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_runs = {name: [] for name in args.configs}
    t_campaign = time.time()

    for seed in range(args.n_seeds):
        for name in args.configs:
            r = run_one(name, seed, args.epochs, args.n_customers, args.batch_size,
                        args.batches_per_epoch, args.device, args.n_test_instances,
                        args.out_dir)
            all_runs[name].append(r)

    # --- Statistiques agrégées ---
    summary = {}
    for name in args.configs:
        dists = [r["test_distance"] for r in all_runs[name]]
        makespans = [r["test_makespan"] for r in all_runs[name]]
        mean_d, ci_d = mean_ci95(dists)
        mean_m, ci_m = mean_ci95(makespans)
        summary[name] = {
            "n_seeds": len(dists),
            "distance_mean": mean_d, "distance_ci95_halfwidth": ci_d,
            "distance_values": dists,
            "makespan_mean": mean_m, "makespan_ci95_halfwidth": ci_m,
            "makespan_values": makespans,
        }
        print(f"\n{name} (n={len(dists)}) : "
              f"distance = {mean_d:.1f} +/- {ci_d:.1f} (IC95%)  |  "
              f"makespan = {mean_m:.1f} +/- {ci_m:.1f} (IC95%)")

    # --- Tests t appariés entre chaque paire de configurations ---
    pairwise = {}
    for i, a in enumerate(args.configs):
        for b in args.configs[i + 1:]:
            da = [r["test_distance"] for r in all_runs[a]]
            db = [r["test_distance"] for r in all_runs[b]]
            if len(da) == len(db) and len(da) >= 2:
                t_stat, p_val = scipy_stats.ttest_rel(da, db)
                key = f"{a}_vs_{b}"
                pairwise[key] = {
                    "t_statistic": float(t_stat), "p_value": float(p_val),
                    "significant_at_0.05": bool(p_val < 0.05),
                    "mean_diff": float(np.mean(da) - np.mean(db)),
                }
                sig = "significatif" if p_val < 0.05 else "non significatif"
                print(f"\n{a} vs {b} (test t apparié, n={len(da)}) : "
                      f"diff moyenne = {pairwise[key]['mean_diff']:+.1f}, "
                      f"p = {p_val:.4f} ({sig})")

    payload = {
        "meta": {
            "configs": args.configs, "n_seeds": args.n_seeds,
            "epochs": args.epochs, "n_customers": args.n_customers,
            "n_test_instances": args.n_test_instances,
            "duration_total_s": time.time() - t_campaign,
        },
        "runs": all_runs,
        "summary": summary,
        "pairwise_tests": pairwise,
    }

    out_path = f"{args.out_dir}/resultats_multi_seed.json"
    with open(out_path, "w") as f:
        json.dump(_to_py(payload), f, indent=2, ensure_ascii=False)
    print(f"\nExporté : {out_path}")


if __name__ == "__main__":
    main()
