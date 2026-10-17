"""
Étape 3 — Entraînement du modèle RL+GNN.

    python scripts/03_train.py
Sur Colab avec GPU : passer device="cuda".
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rl.train import train

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batches-per-epoch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="outputs/checkpoints/policy.pt")
    p.add_argument("--no-domain-randomization", action="store_true",
                    help="Désactive la randomisation de domaine (entraîne sur une config fixe)")
    args, _ = p.parse_known_args()   # tolère les arguments injectés par Colab

    policy, history = train(
        n_customers=args.n_customers, epochs=args.epochs,
        batch_size=args.batch_size, batches_per_epoch=args.batches_per_epoch,
        lr=args.lr, device=args.device, seed=args.seed,
        checkpoint_path=args.out, log_every=1,
        domain_randomization=not args.no_domain_randomization,
    )
    print(f"\nModèle sauvegardé : {args.out}")
    return policy, history

if __name__ == "__main__":
    main()
