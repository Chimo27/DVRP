"""Latence d'inférence par instance isolée (lot de 1), révision du rapport.

Mesure, sur CPU, le temps d'un épisode complet et le temps moyen par décision
pour une politique entraînée, en régime clairvoyant (encodage unique) et en
information partielle (ré-encodage à chaque apparition). Pour comparaison,
le temps amorti sur un lot de 32 est aussi mesuré.
Écrit outputs/dynamic/latence.json.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
torch.set_num_threads(2)
from rl.train import load_checkpoint
from rl.policy import DVRPPolicy
from rl.env import DVRPEnv
from data.generator import generate_dataset

def load(path):
    from rl.train import load_checkpoint
    p, _ = load_checkpoint(path); p.eval(); return p

models = {"clairvoyant": load("outputs/dynamic/clairvoyant_s0.pt"),
          "en_ligne": load("outputs/dynamic/online_s0.pt")}
out = {"device": "cpu", "threads": torch.get_num_threads(), "results": {}}
with torch.no_grad():
    for n in (10, 20, 50, 100):
        insts = generate_dataset(20, base_seed=40_000_000, n_customers=n)
        for name, pol in models.items():
            pol.rollout(DVRPEnv(insts[:1], information=pol.information), greedy=True)          # échauffement
            ep, dec, enc, steps = [], [], [], []
            for inst in insts:
                env = DVRPEnv([inst], information=pol.information)
                t0 = time.perf_counter(); pol.rollout(env, greedy=True); dt = time.perf_counter() - t0
                ep.append(dt * 1000); steps.append(env.n_steps); enc.append(pol.n_encodings)
                dec.append(dt * 1000 / max(env.n_steps, 1))
            r = {"episode_ms_moy": float(np.mean(ep)), "episode_ms_max": float(np.max(ep)),
                 "ms_par_decision_moy": float(np.mean(dec)), "pas_moy": float(np.mean(steps)),
                 "encodages_moy": float(np.mean(enc))}
            if name == "clairvoyant":
                big = generate_dataset(32, base_seed=40_000_000, n_customers=n)
                t0 = time.perf_counter(); pol.rollout(DVRPEnv(big, information=pol.information), greedy=True)
                r["lot32_ms_par_instance"] = (time.perf_counter() - t0) * 1000 / 32
            out["results"].setdefault(str(n), {})[name] = r
            print(n, name, {k: round(v, 2) for k, v in r.items()}, flush=True)
json.dump(out, open("outputs/dynamic/latence.json", "w"), indent=1)
