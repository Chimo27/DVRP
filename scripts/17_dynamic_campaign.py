"""
Campagne du problème dynamique (information en ligne).

Variantes (même budget, mêmes graines, mêmes instances d'évaluation) :
    online      : la politique n'observe que les requêtes apparues  [méthode]
    clairvoyant : la politique observe toutes les requêtes dès t=0   [ablation]
Références : plus proche voisin (PPV), réoptimisation à chaque événement
(REOPT, OR-Tools), aléatoire ; optimum du CVRP statique (borne inférieure de
la distance) sur 90 petites instances.

Configuration : 20 clients, DoD 0,5, congestion désactivée, objectif makespan,
EP époques x 10 lots x 64 instances (CPU). Toutes les mesures sont conservées
instance par instance dans outputs/dynamic/resultats.json.

    EP=200 python scripts/17_dynamic_campaign.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

torch.set_num_threads(2)
from data.generator import generate_dataset
from rl.baselines import nearest_neighbor_policy, random_policy
from rl.env import DVRPEnv
from rl.reopt import reoptimization_policy
from rl.train import save_checkpoint, train

OUT = "outputs/dynamic"
os.makedirs(OUT, exist_ok=True)
PATH = f"{OUT}/resultats.json"
EPOCHS = int(os.environ.get("EP", 200))
BPE, BS = 10, 64
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ENVKW = dict(congestion_mode="off", objective="time", seed=1)
TEST = 20_000_000

# jeux d'évaluation : (nom, nb instances, base de graine, surcharges)
EVALS = [("test", 256, TEST, dict(n_customers=20, degree_of_dynamism=0.5))]
EVALS += [(f"dod{d}", 128, TEST + 1000, dict(n_customers=20, degree_of_dynamism=d))
          for d in (0.0, 0.25, 0.75, 1.0)]
EVALS += [(f"n{n}", 128, TEST + 2000, dict(n_customers=n, degree_of_dynamism=0.5)) for n in (10, 50)]
EVALS += [(f"exact{n}", 30, TEST + n, dict(n_customers=n)) for n in (8, 10, 12)]


def f(t):
    return [round(float(v), 3) for v in t]


def per_instance(metrics):
    return {"dist": f(metrics["distance"]), "mk": f(metrics["makespan"]),
            "wait": f(metrics["wait"]), "unserved": f(metrics["unserved"])}


res = json.load(open(PATH)) if os.path.exists(PATH) else {}
res["config"] = dict(epochs=EPOCHS, batches_per_epoch=BPE, batch_size=BS, n_customers=20,
                     degree_of_dynamism=0.5, congestion_mode="off", objective="time",
                     device="cpu", torch=torch.__version__)

# --- références (calculées une fois) ---
if "baselines" not in res:
    res["baselines"] = {}
for name, k, base, ov in EVALS:
    if name in res["baselines"]:
        continue
    insts = generate_dataset(k, base_seed=base, **ov)
    t0 = time.time()
    _, mn = nearest_neighbor_policy(DVRPEnv(insts, **ENVKW))
    torch.manual_seed(0)
    _, mr = random_policy(DVRPEnv(insts, **ENVKW))
    _, mo = reoptimization_policy(DVRPEnv(insts, **ENVKW), time_limit=0.2)
    res["baselines"][name] = {"nn": per_instance(mn), "random": per_instance(mr),
                              "reopt": per_instance(mo), "reopt_solves": mo["n_solves"],
                              "reopt_s": time.time() - t0,
                              "dod_real": [float(i.is_dynamic[1:].mean()) for i in insts]}
    json.dump(res, open(PATH, "w"))
    b = res["baselines"][name]
    print(f"[réf] {name}: PPV {np.mean(b['nn']['dist']):.1f} | REOPT {np.mean(b['reopt']['dist']):.1f} "
          f"| aléatoire {np.mean(b['random']['dist']):.1f} [{b['reopt_s']:.0f} s]", flush=True)

# --- entraînements ---
for seed in SEEDS:
    for info in ("online", "clairvoyant"):
        key = f"{info}_s{seed}"
        if key in res:
            continue
        t0 = time.time()
        pol, hist = train(n_customers=20, epochs=EPOCHS, batch_size=BS, batches_per_epoch=BPE,
                          device="cpu", seed=seed, domain_randomization=False, log_every=25,
                          verbose=True, congestion_mode="off", objective="time",
                          policy_kwargs={"information": info})
        rec = {"information": info, "seed": seed, "train_s": time.time() - t0,
               "baseline_updates": int(sum(h["baseline_updated"] for h in hist)),
               "history": [{k: h[k] for k in ("epoch", "eval_distance", "train_distance",
                                               "gap_vs_nn_pct")} for h in hist],
               "eval": {}}
        pol.eval()
        for name, k, base, ov in EVALS:
            insts = generate_dataset(k, base_seed=base, **ov)
            with torch.no_grad():
                _, _, _, m = pol.rollout(DVRPEnv(insts, information=info, **ENVKW), greedy=True)
            rec["eval"][name] = per_instance(m)
            rec["eval"][name]["n_encodings"] = pol.n_encodings
        save_checkpoint(pol, hist, f"{OUT}/{key}.pt")
        res[key] = rec
        json.dump(res, open(PATH, "w"))
        rl = np.mean(rec["eval"]["test"]["dist"])
        nn = np.mean(res["baselines"]["test"]["nn"]["dist"])
        ro = np.mean(res["baselines"]["test"]["reopt"]["dist"])
        print(f"== {key}: RL {rl:.1f} | PPV {nn:.1f} ({(rl-nn)/nn*100:+.1f}%) | "
              f"REOPT {ro:.1f} ({(rl-ro)/ro*100:+.1f}%) [{rec['train_s']/60:.1f} min]", flush=True)
print("TERMINE", flush=True)
