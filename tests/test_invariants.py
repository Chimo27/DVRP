"""
Tests automatisés des invariants du projet.

    python -m pytest tests/ -v
    python tests/test_invariants.py     (si pytest n'est pas installé)

Ces tests vérifient les propriétés dont dépendent les conclusions du
rapport. Trois d'entre eux correspondent directement à des défaillances
réellement rencontrées pendant le développement (cf. chapitre « Difficultés
méthodologiques rencontrées ») : ils existent pour empêcher leur réapparition
silencieuse.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from data.generator import generate_instance, generate_dataset
from data.validation import validate_instance
from data.travel_time import compute_travel_time, env_time_to_clock
from rl.env import DVRPEnv
from rl.policy import DVRPPolicy
from rl.baselines import nearest_neighbor_policy, random_policy


# =====================================================================
# Génération d'instances
# =====================================================================

def test_reproductibilite_meme_graine():
    """Deux générations avec la même graine doivent être identiques."""
    a = generate_instance(seed=42, n_customers=15)
    b = generate_instance(seed=42, n_customers=15)
    assert np.allclose(a.locations, b.locations)
    assert np.allclose(a.demands, b.demands)
    assert np.allclose(a.appearance_times, b.appearance_times)


def test_graines_differentes_donnent_instances_differentes():
    a = generate_instance(seed=1, n_customers=15)
    b = generate_instance(seed=2, n_customers=15)
    assert not np.allclose(a.locations, b.locations)


def test_depot_sans_demande():
    inst = generate_instance(seed=0, n_customers=10)
    assert inst.demands[0] == 0
    assert inst.appearance_times[0] == 0
    assert not inst.is_dynamic[0]


def test_demande_jamais_superieure_capacite():
    """Invariant critique : garantit la terminaison des épisodes."""
    for seed in range(20):
        inst = generate_instance(seed=seed, n_customers=25)
        assert inst.demands.max() <= inst.capacity


def test_generation_refuse_configuration_infaisable():
    """Une demande maximale supérieure à la capacité doit lever une erreur."""
    with pytest.raises(ValueError):
        generate_instance(seed=0, n_customers=10, capacity=5,
                          demand_range=(1, 20))


def test_matrice_distances_symetrique_diagonale_nulle():
    inst = generate_instance(seed=7, n_customers=20)
    d = inst.dist_matrix
    assert np.allclose(d, d.T, atol=1e-9)
    assert np.allclose(np.diag(d), 0.0, atol=1e-9)


def test_instants_revelation_dans_horizon():
    inst = generate_instance(seed=3, n_customers=30)
    horizon = inst.metadata["planning_horizon"][1]
    assert inst.appearance_times.min() >= 0
    assert inst.appearance_times.max() <= horizon


def test_clients_statiques_ont_revelation_nulle():
    inst = generate_instance(seed=5, n_customers=30)
    statiques = ~inst.is_dynamic
    assert np.allclose(inst.appearance_times[statiques], 0.0)


def test_validateur_ne_signale_rien_sur_instances_valides():
    for seed in range(10):
        inst = generate_instance(seed=seed, n_customers=20)
        _, violations = validate_instance(inst)
        assert violations == [], f"violations sur la graine {seed} : {violations}"


# =====================================================================
# Environnement — invariants de sûreté
# =====================================================================

def test_masque_jamais_entierement_vide():
    """RÉGRESSION : un masque vide produisait un NaN contaminant tout le lot."""
    insts = generate_dataset(32, base_seed=0, n_customers=15)
    env = DVRPEnv(insts, seed=0)
    state = env.reset()
    for _ in range(env.max_steps):
        if env.done.all():
            break
        assert state["mask"].sum(dim=1).min().item() > 0, \
            "au moins une ligne de masque est entièrement False"
        probs = state["mask"].float()
        action = torch.multinomial(probs / probs.sum(dim=1, keepdim=True), 1).squeeze(-1)
        state, _ = env.step(action)


def test_terminaison_garantie():
    """Tout épisode se termine avant le garde-fou max_steps."""
    for cong in ["off", "multiplicative"]:
        insts = generate_dataset(32, base_seed=1, n_customers=20)
        env = DVRPEnv(insts, congestion_mode=cong, seed=0)
        state = env.reset()
        steps = 0
        while not env.done.all() and steps < env.max_steps:
            probs = state["mask"].float()
            action = torch.multinomial(probs / probs.sum(dim=1, keepdim=True), 1).squeeze(-1)
            state, _ = env.step(action)
            steps += 1
        assert env.done.all(), f"épisode non terminé (congestion={cong})"
        assert steps < env.max_steps


def test_tous_clients_servis_a_la_fin():
    """RÉGRESSION : le modèle avait appris à ne jamais se déplacer, faute de
    pénalité sur les clients non visités."""
    insts = generate_dataset(16, base_seed=2, n_customers=15)
    env = DVRPEnv(insts, seed=0)
    _, metrics = nearest_neighbor_policy(env)
    assert metrics["unserved"].sum().item() == 0
    assert metrics["completed"].mean().item() == 1.0


def test_capacite_jamais_depassee():
    insts = generate_dataset(16, base_seed=3, n_customers=20)
    env = DVRPEnv(insts, seed=0)
    state = env.reset()
    for _ in range(env.max_steps):
        if env.done.all():
            break
        assert (env.cur_cap >= -1e-6).all(), "capacité résiduelle négative"
        probs = state["mask"].float()
        action = torch.multinomial(probs / probs.sum(dim=1, keepdim=True), 1).squeeze(-1)
        state, _ = env.step(action)


def test_client_jamais_servi_avant_revelation():
    """Invariant central du cas dynamique."""
    insts = generate_dataset(16, base_seed=4, n_customers=20)
    env = DVRPEnv(insts, seed=0)
    state = env.reset()
    idx = torch.arange(env.batch_size)
    for _ in range(env.max_steps):
        if env.done.all():
            break
        probs = state["mask"].float()
        action = torch.multinomial(probs / probs.sum(dim=1, keepdim=True), 1).squeeze(-1)
        appear = env.appear[idx, action]
        active = ~env.done
        # Tolérance numérique : le temps courant doit avoir atteint r_i
        assert ((env.cur_time + 1e-6 >= appear) | ~active).all(), \
            "un client a été sélectionné avant sa révélation"
        state, _ = env.step(action)


def test_reproductibilite_environnement():
    """Deux environnements de même graine doivent produire les mêmes résultats."""
    insts = generate_dataset(16, base_seed=5, n_customers=15)
    e1 = DVRPEnv(insts, congestion_mode="multiplicative", seed=7)
    e2 = DVRPEnv(insts, congestion_mode="multiplicative", seed=7)
    _, m1 = nearest_neighbor_policy(e1)
    _, m2 = nearest_neighbor_policy(e2)
    assert torch.allclose(m1["makespan"], m2["makespan"])


def test_congestion_reproductible_entre_deux_rollouts():
    """RÉGRESSION : sans réinitialisation du générateur, politique et baseline
    subissaient des congestions différentes, faussant la comparaison."""
    insts = generate_dataset(16, base_seed=6, n_customers=15)
    env = DVRPEnv(insts, congestion_mode="multiplicative", seed=3)
    _, m1 = nearest_neighbor_policy(env)
    _, m2 = nearest_neighbor_policy(env)
    assert torch.allclose(m1["makespan"], m2["makespan"])


# =====================================================================
# Modèle de temps de trajet
# =====================================================================

def test_congestion_off_equivaut_distance_sur_vitesse():
    d = torch.tensor([10.0, 50.0, 100.0])
    clock = torch.zeros(3)
    tt = compute_travel_time(d, clock, mode="off", velocity=2.0)
    assert torch.allclose(tt, d / 2.0)


def test_multiplicative_ralentit_en_pointe():
    """Le trajet doit coûter plus cher en heure de pointe qu'en heure creuse."""
    d = torch.full((2000,), 50.0)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    peak = compute_travel_time(d, env_time_to_clock(torch.zeros(2000)),
                               mode="multiplicative", generator=g1,
                               enable_accidents=False)
    off = compute_travel_time(d, env_time_to_clock(torch.full((2000,), 240.0)),
                              mode="multiplicative", generator=g2,
                              enable_accidents=False)
    assert peak.mean() > off.mean() * 1.3


def test_faithful_effet_negligeable():
    """Documente l'incohérence d'unités analysée dans le rapport : en minutes,
    le modèle d'origine ne produit pas d'effet observable."""
    d = torch.full((2000,), 50.0)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    peak = compute_travel_time(d, env_time_to_clock(torch.zeros(2000)),
                               mode="faithful", generator=g1, enable_accidents=False)
    off = compute_travel_time(d, env_time_to_clock(torch.full((2000,), 240.0)),
                              mode="faithful", generator=g2, enable_accidents=False)
    assert abs(peak.mean() / off.mean() - 1.0) < 0.05


# =====================================================================
# Politique et entraînement
# =====================================================================

def test_politique_produit_gradients_finis():
    """RÉGRESSION : l'entropie produisait des NaN via 0 x log(0)."""
    insts = generate_dataset(8, base_seed=7, n_customers=12)
    env = DVRPEnv(insts, congestion_mode="multiplicative", seed=0)
    policy = DVRPPolicy(embed_dim=32, n_heads=4, n_layers=2)
    log_probs, reward, entropy, _ = policy.rollout(env, greedy=False)
    loss = -(reward.detach() * log_probs).mean() - 0.01 * entropy.mean()
    loss.backward()
    for name, p in policy.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"gradient non fini : {name}"


def test_decodage_glouton_deterministe():
    insts = generate_dataset(8, base_seed=8, n_customers=12)
    policy = DVRPPolicy(embed_dim=32, n_heads=4, n_layers=2)
    with torch.no_grad():
        e1 = DVRPEnv(insts, seed=0)
        _, r1, _, _ = policy.rollout(e1, greedy=True)
        e2 = DVRPEnv(insts, seed=0)
        _, r2, _, _ = policy.rollout(e2, greedy=True)
    assert torch.allclose(r1, r2)


def test_politique_generalise_a_taille_non_vue():
    """Un encodeur sur graphe doit accepter n'importe quelle taille d'entrée."""
    policy = DVRPPolicy(embed_dim=32, n_heads=4, n_layers=2)
    for n in [5, 20, 40]:
        insts = generate_dataset(4, base_seed=9, n_customers=n)
        env = DVRPEnv(insts, seed=0)
        with torch.no_grad():
            _, reward, _, metrics = policy.rollout(env, greedy=True)
        assert torch.isfinite(reward).all()
        assert metrics["unserved"].sum().item() == 0


def test_objectifs_coherents():
    """L'objectif 'time' doit produire une récompense égale à -makespan."""
    insts = generate_dataset(8, base_seed=10, n_customers=12)
    env = DVRPEnv(insts, objective="time", seed=0)
    reward, metrics = nearest_neighbor_policy(env)
    assert torch.allclose(reward, -metrics["makespan"], atol=1e-4)

    env = DVRPEnv(insts, objective="distance", seed=0)
    reward, metrics = nearest_neighbor_policy(env)
    assert torch.allclose(reward, -metrics["distance"], atol=1e-4)


def test_heuristique_plus_proche_voisin_bat_aleatoire():
    insts = generate_dataset(32, base_seed=11, n_customers=20)
    r_nn, _ = nearest_neighbor_policy(DVRPEnv(insts, seed=0))
    r_rd, _ = random_policy(DVRPEnv(insts, seed=0))
    assert r_nn.mean() > r_rd.mean()


# =====================================================================
if __name__ == "__main__":
    # Exécution sans pytest : lance chaque test et rapporte
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    echecs = []
    for name, fn in tests:
        try:
            fn()
            print(f"  OK    {name}")
        except Exception as e:
            print(f"  ECHEC {name} : {type(e).__name__}: {e}")
            echecs.append(name)
    print(f"\n{len(tests) - len(echecs)}/{len(tests)} tests passés")
    sys.exit(1 if echecs else 0)
