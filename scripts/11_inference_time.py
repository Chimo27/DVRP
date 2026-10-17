"""
Étape 11 — Mesure du temps d'inférence.

    python scripts/11_inference_time.py --device cpu

Le rapport affirme que produire une décision se réduit à « une inférence du
réseau, dont le coût est de l'ordre de la milliseconde ». Cette affirmation
justifie l'approche entière face aux métaheuristiques ; elle doit donc être
mesurée, non postulée.

Trois grandeurs distinctes sont mesurées, souvent confondues :

  1. ENCODAGE : coût de l'encodeur GNN, payé UNE FOIS par instance ;
  2. DÉCISION : coût d'un pas de décodage, payé à CHAQUE choix de nœud ;
                c'est cette grandeur qui doit être de l'ordre de la
                milliseconde pour justifier l'usage en temps réel ;
  3. ÉPISODE  : coût total de la résolution complète d'une instance.

NOTE SUR LES POIDS : le coût d'une passe avant ne dépend pas de la valeur des
poids, seulement de l'architecture et de la taille de l'entrée. La mesure est
donc valide avec un modèle non entraîné, ce qui évite d'exiger un point de
reprise. Le script accepte néanmoins --checkpoint.
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
from rl.policy import DVRPPolicy
from rl.baselines import nearest_neighbor_policy


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure(policy, n_customers, device, n_instances=32, n_repeat=6,
            congestion_mode="multiplicative", objective="time"):
    insts = generate_dataset(n_instances, base_seed=30_000_000 + n_customers,
                             n_customers=n_customers)
    policy.eval()

    # Préchauffage
    env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                  objective=objective, seed=0)
    state = env.reset()
    for _ in range(2):
        h = policy.encode(state)
        _ = policy.decode_step(h, state)
    _sync(device)

    # 1. Encodage
    enc = []
    for _ in range(n_repeat):
        env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                      objective=objective, seed=0)
        state = env.reset()
        _sync(device); t0 = time.perf_counter()
        h = policy.encode(state)
        _sync(device)
        enc.append((time.perf_counter() - t0) / n_instances * 1000)

    # 2. Un pas de décision
    env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                  objective=objective, seed=0)
    state = env.reset()
    h = policy.encode(state)
    dec = []
    for _ in range(n_repeat):
        _sync(device); t0 = time.perf_counter()
        _ = policy.decode_step(h, state)
        _sync(device)
        dec.append((time.perf_counter() - t0) / n_instances * 1000)

    # 3. Épisode complet
    epi, steps = [], []
    for _ in range(max(2, n_repeat // 3)):
        env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                      objective=objective, seed=0)
        _sync(device); t0 = time.perf_counter()
        policy.rollout(env, greedy=True)
        _sync(device)
        epi.append((time.perf_counter() - t0) / n_instances * 1000)
        steps.append(env.n_steps)

    # 4. Référence heuristique
    nn = []
    for _ in range(max(2, n_repeat // 3)):
        env = DVRPEnv(insts, device=device, congestion_mode=congestion_mode,
                      objective=objective, seed=0)
        _sync(device); t0 = time.perf_counter()
        nearest_neighbor_policy(env)
        _sync(device)
        nn.append((time.perf_counter() - t0) / n_instances * 1000)

    return {
        "n_customers": n_customers,
        "n_instances_par_lot": n_instances,
        "encodage_ms": float(np.mean(enc)),
        "decision_ms": float(np.mean(dec)),
        "episode_ms": float(np.mean(epi)),
        "pas_par_episode": float(np.mean(steps)),
        "nn_episode_ms": float(np.mean(nn)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--n-instances", type=int, default=32)
    p.add_argument("--n-repeat", type=int, default=6)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="outputs/final/temps_inference.json")
    args, _ = p.parse_known_args()

    if args.checkpoint and os.path.exists(args.checkpoint):
        from rl.train import load_checkpoint
        policy, _ = load_checkpoint(args.checkpoint, device=args.device)
        source = f"modèle entraîné ({args.checkpoint})"
    else:
        policy = DVRPPolicy().to(args.device)
        source = "modèle non entraîné (coût indépendant des poids)"

    n_params = sum(q.numel() for q in policy.parameters())
    print(f"Device : {args.device} | {source}")
    print(f"Paramètres : {n_params:,}".replace(",", " "))
    print()
    print(f"{'Clients':>8} {'Encodage':>10} {'Décision':>11} {'Épisode':>10} "
          f"{'Pas':>6} {'PPV':>8}")
    print(f"{'':>8} {'(ms)':>10} {'(ms/pas)':>11} {'(ms)':>10} {'':>6} {'(ms)':>8}")
    print("-" * 58)

    results = {}
    for n in args.sizes:
        r = measure(policy, n, args.device, args.n_instances, args.n_repeat)
        results[str(n)] = r
        print(f"{n:>8} {r['encodage_ms']:>10.2f} {r['decision_ms']:>11.3f} "
              f"{r['episode_ms']:>10.2f} {r['pas_par_episode']:>6.0f} "
              f"{r['nn_episode_ms']:>8.2f}")

    payload = {
        "meta": {
            "device": args.device, "source_modele": source,
            "n_parametres": n_params,
            "n_instances_par_lot": args.n_instances,
            "note": ("Temps PAR INSTANCE, mesures sur un lot traite en "
                     "parallele (mode nominal de l'environnement vectorise). "
                     "Une inference sur instance isolee serait plus lente par "
                     "instance, la parallelisation n'etant pas exploitee."),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nExporté : {args.out}")


if __name__ == "__main__":
    main()
