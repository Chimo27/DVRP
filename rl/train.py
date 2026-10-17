"""
Entraînement REINFORCE avec baseline gloutonne FIGÉE.

POURQUOI UNE BASELINE FIGÉE
---------------------------
Une version précédente recalculait la baseline à chaque batch avec la
politique COURANTE :

    advantage = reward(échantillonné, π_courante) - reward(glouton, π_courante)

Problème observé en pratique : à mesure que π devient déterministe, le
rollout échantillonné se rapproche du rollout glouton, l'avantage tend vers
zéro, le gradient s'évanouit et l'apprentissage se fige. C'est exactement ce
qu'on a mesuré : reward -989 -> -814 sur 100 époques, puis quasi plat
(-814 -> -808) sur les 100 suivantes.

Solution (Kool, van Hoof & Welling, 2019) : la baseline est une COPIE GELÉE
de la politique. Elle n'est remplacée par la politique courante que si
celle-ci est significativement meilleure sur un jeu d'évaluation fixe
(test t apparié, p < seuil). L'avantage garde ainsi un signal non nul :

    advantage = reward(échantillonné, π_courante) - reward(glouton, π_figée)

RANDOMISATION DE DOMAINE
-------------------------
Si domain_randomization=True (par défaut), chaque batch d'entraînement
pioche une configuration d'instance différente (taille, distribution
spatiale, degré de dynamisme, demande, capacité) au lieu de toujours voir
la même. Objectif : une politique robuste sur toute une famille de
problèmes plutôt que sur-spécialisée à une configuration unique.

Le jeu d'ÉVALUATION reste volontairement figé sur la configuration
canonique (INSTANCE_CONFIG) pour permettre un suivi cohérent du progrès
d'une époque à l'autre — seul l'entraînement varie.

DIAGNOSTIC
----------
distance, clients non servis et makespan sont journalisés SÉPARÉMENT. Suivre
uniquement la récompense agrégée masque d'où vient (ou ne vient pas) le
progrès.
"""

import os
import sys
import time
import copy

import numpy as np
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TRAIN_CONFIG, INSTANCE_CONFIG
from data.generator import generate_dataset
from data.domain_randomization import sample_domain_config
from rl.env import DVRPEnv
from rl.policy import DVRPPolicy
from rl.baselines import nearest_neighbor_policy


# ---------------------------------------------------------------------------
def _make_env(n_instances, seed, device, instance_overrides, congestion_mode=None,
              objective=None, information="online"):
    insts = generate_dataset(n_instances, base_seed=seed, **instance_overrides)
    # seed de congestion dérivée de la seed d'instances : le rollout de la
    # politique et celui de la baseline partagent ainsi la même réalisation
    # de congestion, condition nécessaire à une comparaison équitable.
    return DVRPEnv(insts, device=device, congestion_mode=congestion_mode, seed=seed,
                    objective=objective, information=information)


@torch.no_grad()
def evaluate(policy, env, greedy=True):
    """Évalue une politique sur un environnement, sans gradient."""
    _, reward, _, metrics = policy.rollout(env, greedy=greedy)
    return reward, metrics


# ---------------------------------------------------------------------------
def train(
    n_customers=None, epochs=None, batch_size=None, batches_per_epoch=None,
    lr=None, device=None, seed=None, entropy_coef=None,
    baseline_update_period=None, baseline_eval_size=None, baseline_significance=None,
    log_every=1, checkpoint_path=None, checkpoint_every=20,
    instance_overrides=None, verbose=True,
    domain_randomization=None, domain_ranges=None, congestion_mode=None,
    objective=None, baseline_mode="frozen", policy_kwargs=None,
):
    cfg = TRAIN_CONFIG
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    batches_per_epoch = batches_per_epoch or cfg["batches_per_epoch"]
    lr = lr or cfg["lr"]
    device = device or cfg["device"]
    seed = cfg["seed"] if seed is None else seed
    entropy_coef = cfg["entropy_coef"] if entropy_coef is None else entropy_coef
    bl_period = baseline_update_period or cfg["baseline_update_period"]
    bl_eval_size = baseline_eval_size or cfg["baseline_eval_size"]
    bl_alpha = baseline_significance or cfg["baseline_significance"]
    domain_randomization = (cfg["domain_randomization"] if domain_randomization is None
                             else domain_randomization)
    from config import ENV_CONFIG
    congestion_mode = (ENV_CONFIG.get("congestion_mode", "off") if congestion_mode is None
                        else congestion_mode)
    objective = ENV_CONFIG.get("objective", "distance") if objective is None else objective

    if baseline_mode not in ("frozen", "moving"):
        raise ValueError("baseline_mode doit être 'frozen' ou 'moving'")

    instance_overrides = dict(instance_overrides or {})
    if n_customers is not None:
        instance_overrides["n_customers"] = n_customers
    n_cust = instance_overrides.get("n_customers", INSTANCE_CONFIG["n_customers"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = DVRPPolicy(**(policy_kwargs or {})).to(device)
    info = policy.information
    baseline = copy.deepcopy(policy).to(device)
    for p in baseline.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Jeu d'évaluation FIXE (seeds réservées, jamais vues à l'entraînement)
    # : sert à décider des mises à jour de baseline et à suivre le progrès
    # réel de façon comparable d'une époque à l'autre.
    EVAL_SEED_BASE = 10_000_000
    eval_env = _make_env(bl_eval_size, EVAL_SEED_BASE, device, instance_overrides, congestion_mode, objective, info)

    # Référence heuristique, calculée une fois pour situer les scores
    nn_reward, nn_metrics = nearest_neighbor_policy(_make_env(
        bl_eval_size, EVAL_SEED_BASE, device, instance_overrides, congestion_mode, objective))
    nn_dist = nn_metrics["distance"].mean().item()

    if verbose:
        print(f"Entraînement DVRP | {n_cust} clients | device={device}")
        print(f"Référence plus proche voisin (jeu d'éval) : distance={nn_dist:.1f}")
        if domain_randomization:
            print("Randomisation de domaine : ACTIVÉE "
                  "(taille, spatial, DoD, demande, capacité varient à chaque batch)")
        print()

    # rng dédié au tirage des configurations de domaine, séparé du hasard
    # de génération d'instance elle-même (deux sources de hasard distinctes,
    # chacune reproductible indépendamment à partir de `seed`).
    domain_rng = np.random.default_rng(seed + 999_983 if seed is not None else None)

    history = []
    n_baseline_updates = 0
    train_seed_counter = 0

    for epoch in range(epochs):
        t0 = time.time()
        ep_loss, ep_reward, ep_dist, ep_unserved = [], [], [], []

        policy.train()
        for _ in range(batches_per_epoch):
            # Seeds d'entraînement disjointes des seeds d'évaluation
            train_seed_counter += batch_size

            if domain_randomization:
                # Config différente à chaque batch -> le modèle ne peut pas
                # se sur-spécialiser sur une seule configuration. Le jeu
                # d'évaluation, lui, reste fixe (cf. plus haut) pour un
                # suivi cohérent du progrès d'une époque à l'autre.
                batch_overrides = sample_domain_config(domain_rng, domain_ranges)
            else:
                batch_overrides = instance_overrides

            env_s = _make_env(batch_size, train_seed_counter, device, batch_overrides, congestion_mode, objective, info)
            env_b = _make_env(batch_size, train_seed_counter, device, batch_overrides, congestion_mode, objective, info)

            log_probs, reward, entropy, metrics = policy.rollout(env_s, greedy=False)
            with torch.no_grad():
                if baseline_mode == "frozen":
                    # Baseline figée (comportement historique, cf. docstring
                    # du module) : rollout glouton d'une copie gelée.
                    _, reward_bl, _, _ = baseline.rollout(env_b, greedy=True)
                else:
                    # Baseline mobile (ablation) : rollout glouton de la
                    # politique COURANTE, recalculée à chaque batch. C'est
                    # cette variante dont on suppose, sans l'avoir vérifié
                    # expérimentalement jusqu'ici, qu'elle fait s'éteindre
                    # le gradient à mesure que la politique devient
                    # déterministe (cf. section « Baseline gloutonne figée »
                    # du rapport). `baseline` (la copie figée) n'est alors
                    # calculée que pour le suivi, jamais utilisée dans la perte.
                    _, reward_bl, _, _ = policy.rollout(env_b, greedy=True)

            advantage = (reward - reward_bl).detach()
            loss = -(advantage * log_probs).mean() - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
            optimizer.step()

            ep_loss.append(loss.item())
            ep_reward.append(reward.mean().item())
            ep_dist.append(metrics["distance"].mean().item())
            ep_unserved.append(metrics["unserved"].mean().item())

        # --- Évaluation gloutonne sur le jeu fixe ---
        policy.eval()
        eval_reward, eval_metrics = evaluate(policy, eval_env, greedy=True)
        eval_dist = eval_metrics["distance"].mean().item()

        # --- Mise à jour éventuelle de la baseline (test t apparié) ---
        updated = False
        if (epoch + 1) % bl_period == 0:
            bl_reward, _ = evaluate(baseline, eval_env, greedy=True)
            diff = (eval_reward - bl_reward).cpu().numpy()
            if diff.mean() > 0 and np.abs(diff).sum() > 0:
                # unilatéral : la politique est-elle significativement meilleure ?
                _, p_two = scipy_stats.ttest_rel(
                    eval_reward.cpu().numpy(), bl_reward.cpu().numpy())
                p_one = p_two / 2
                if p_one < bl_alpha:
                    baseline.load_state_dict(policy.state_dict())
                    for p in baseline.parameters():
                        p.requires_grad_(False)
                    n_baseline_updates += 1
                    updated = True

        rec = {
            "epoch": epoch,
            "loss": float(np.mean(ep_loss)),
            "train_reward": float(np.mean(ep_reward)),
            "train_distance": float(np.mean(ep_dist)),
            "train_unserved": float(np.mean(ep_unserved)),
            "eval_distance": eval_dist,
            "eval_reward": eval_reward.mean().item(),
            "baseline_updated": updated,
            "gap_vs_nn_pct": (eval_dist - nn_dist) / nn_dist * 100,
            "time_s": time.time() - t0,
        }
        history.append(rec)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            flag = " *baseline maj" if updated else ""
            print(f"Ép. {epoch:4d} | éval dist={eval_dist:7.1f} "
                  f"(vs NN {rec['gap_vs_nn_pct']:+6.1f}%) | "
                  f"train dist={rec['train_distance']:7.1f} | "
                  f"non servis={rec['train_unserved']:.2f} | "
                  f"{rec['time_s']:.1f}s{flag}")

        if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(policy, history, checkpoint_path)

    if verbose:
        print(f"\nMises à jour de baseline : {n_baseline_updates}")

    if checkpoint_path:
        save_checkpoint(policy, history, checkpoint_path)

    return policy, history


# ---------------------------------------------------------------------------
def save_checkpoint(policy, history, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "history": history,
                "policy_config": {"information": policy.information,
                                  "use_edge_attr": policy.encoder.use_edge_attr}}, path)


def load_checkpoint(path, device="cpu", **policy_kwargs):
    """Recharge une politique. Le régime d'information et la présence des
    attributs d'arêtes sont lus dans le point de reprise s'ils y figurent ;
    sinon (anciens points de reprise), ils doivent être passés en argument."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Les points de reprise antérieurs à l'ajout du régime en ligne ne
    # contiennent pas de configuration : ils proviennent tous de modèles
    # clairvoyants (campagne initiale), d'où ce défaut.
    default = {"information": "clairvoyant"} if "policy_config" not in ckpt else {}
    cfg = {**default, **ckpt.get("policy_config", {}), **policy_kwargs}
    policy = DVRPPolicy(**cfg).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    return policy, ckpt.get("history", [])
