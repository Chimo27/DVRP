"""
Étape 4 — Évaluation du modèle entraîné face aux heuristiques.

    python scripts/04_evaluate.py --checkpoint outputs/checkpoints/policy.pt
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rl.train import load_checkpoint
from rl.evaluate import evaluate_all, evaluate_generalization, evaluate_domain_robustness

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="outputs/checkpoints/policy.pt")
    p.add_argument("--n-customers", type=int, default=20)
    p.add_argument("--n-instances", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = p.parse_known_args()

    policy, history = load_checkpoint(args.checkpoint, device=args.device)

    print(f"\n--- Évaluation ({args.n_customers} clients, {args.n_instances} instances) ---\n")
    evaluate_all(policy, n_instances=args.n_instances, device=args.device,
                 instance_overrides={"n_customers": args.n_customers})

    print("\n--- Généralisation à d'autres tailles ---\n")
    evaluate_generalization(policy, sizes=(10, 20, 50), n_instances=64, device=args.device)

    print("\n--- Robustesse à travers plusieurs configurations de domaine ---\n")
    evaluate_domain_robustness(policy, n_instances=64, device=args.device)

if __name__ == "__main__":
    main()
