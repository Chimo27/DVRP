"""
Tests prouvant que le problème traité est réellement dynamique : la politique
ne dispose jamais d'information sur les requêtes futures, et les événements
(apparitions, service, capacité, attente, fin d'épisode) sont simulés
correctement.

A  Invariance de l'observation et des décisions à l'information cachée
B  Révélation progressive (jamais avant r_i, exactement une fois, jamais retirée)
C  Service unique, capacité, recharge au dépôt
D  Attente au dépôt sans exposition préalable du prochain instant d'apparition
E  Terminaison
F  Parité entraînement / évaluation / inférence, cohérence des régimes
G  Cas limites (seul le dépôt visible, une seule requête visible)
H  Le régime clairvoyant (ablation) dépend bien de l'information cachée
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.generator import generate_dataset, generate_instance
from data.instance import Instance
from rl.baselines import nearest_neighbor_policy
from rl.env import DVRPEnv
from rl.infer import solve_instance
from rl.policy import DVRPPolicy

T_SPLIT = 120.0


# ---------------------------------------------------------------------------
def _make_instance(locs, dem, app, srv, cap=30):
    locs = np.asarray(locs, float)
    d = locs[:, None, :] - locs[None, :, :]
    return Instance(locations=locs, demands=np.asarray(dem, float), capacity=cap,
                    appearance_times=np.asarray(app, float),
                    is_dynamic=np.asarray(app, float) > 0,
                    service_times=np.asarray(srv, float),
                    dist_matrix=np.sqrt((d ** 2).sum(-1)),
                    metadata={"velocity": 1.0, "planning_horizon": (0, 480)})


def _pair(seed, n=20, extra=5):
    """Deux instances identiques pour tout ce qui apparaît avant T_SPLIT, mais
    avec des requêtes futures différentes (positions, demandes, durées,
    instants > T_SPLIT) et un nombre de requêtes futures différent."""
    base = generate_instance(seed=seed, n_customers=n)
    rng = np.random.default_rng(seed + 99)
    keep = [0] + [i for i in range(1, n + 1) if base.appearance_times[i] < T_SPLIT]

    def build(k_hidden):
        locs = [base.locations[i] for i in keep]
        dem = [base.demands[i] for i in keep]
        app = [base.appearance_times[i] for i in keep]
        srv = [base.service_times[i] for i in keep]
        for _ in range(k_hidden):
            locs.append(rng.uniform(0, 100, 2)); dem.append(float(rng.integers(1, 10)))
            app.append(float(rng.uniform(T_SPLIT + 1, 360))); srv.append(float(rng.integers(5, 16)))
        return _make_instance(locs, dem, app, srv, cap=base.capacity)

    k = n + 1 - len(keep)
    return build(k), build(k + extra), len(keep)


def _visible_part(obs, k):
    """Composantes de l'observation restreintes aux k premiers nœuds (visibles)."""
    out = {}
    for key, v in obs.items():
        if not torch.is_tensor(v):
            out[key] = v
        elif v.dim() == 3 and v.shape[1] == v.shape[2]:
            out[key] = v[:, :k, :k]
        elif v.dim() >= 2 and v.shape[1] >= k and key not in ("loc",):
            out[key] = v[:, :k]
        elif key == "loc":
            out[key] = v[:, :k]
        else:
            out[key] = v
    return out


def _greedy_trace(policy, inst, until):
    """Décisions gloutonnes et scores tant que le temps courant est < until."""
    env = DVRPEnv([inst], information=policy.information)
    env.reset()
    obs = env.observe()
    h = policy.encode(obs)
    vis = obs["visible"].clone()
    acts, scores, observations = [], [], []
    for _ in range(env.max_steps):
        if env.done.all() or float(env.cur_time[0]) >= until:
            break
        if (obs["visible"] != vis).any():
            h = policy.encode(obs); vis = obs["visible"].clone()
        lg = policy.decode_step(h, obs)
        a = lg.argmax(-1)
        acts.append(int(a)); scores.append(lg[0].clone()); observations.append(obs)
        env.step(a)
        obs = env.observe()
    return acts, scores, observations


# ---------------------------------------------------------------------------
# A. Invariance
@pytest.mark.parametrize("extra", [0, 5])
def test_A_observation_identique(extra):
    for seed in range(5):
        a, b, k = _pair(500 + seed, extra=extra)
        oa, ob = DVRPEnv([a]).observe(), DVRPEnv([b]).observe()
        va, vb = _visible_part(oa, k), _visible_part(ob, k)
        for key in va:
            if torch.is_tensor(va[key]):
                assert torch.equal(va[key], vb[key]), key
        # aucune trace de requête cachée hors de la partie visible
        for o in (oa, ob):
            assert not o["visible"][0, k:].any()
            assert float(o["loc"][0, k:].abs().sum()) == 0.0
            assert float(o["demand"][0, k:].abs().sum()) == 0.0
            assert float(o["appear"][0, k:].abs().sum()) == 0.0
            assert float(o["service"][0, k:].abs().sum()) == 0.0
            assert float(o["dist"][0, k:, :].abs().sum()) == 0.0
            assert float(o["dist"][0, :, k:].abs().sum()) == 0.0
            assert not o["mask"][0, k:].any()


@pytest.mark.parametrize("extra", [0, 5])
def test_A_decisions_identiques_avant_revelation(extra):
    torch.manual_seed(0)
    pol = DVRPPolicy(information="online").eval()
    n_dec = 0
    with torch.no_grad():
        for seed in range(10):
            a, b, k = _pair(700 + seed, extra=extra)
            ta, sa, _ = _greedy_trace(pol, a, T_SPLIT)
            tb, sb, _ = _greedy_trace(pol, b, T_SPLIT)
            assert ta == tb
            for x, y in zip(sa, sb):
                assert torch.allclose(x[:k], y[:k], atol=1e-6, rtol=0)
            n_dec += len(ta)
    assert n_dec > 10            # le test porte bien sur des décisions


def test_A_independance_au_lot():
    """Les décisions pour une instance ne dépendent pas des autres instances
    du lot (encodage par instance)."""
    torch.manual_seed(0)
    pol = DVRPPolicy(information="online").eval()
    insts = generate_dataset(6, base_seed=31_000, n_customers=15)
    with torch.no_grad():
        env = DVRPEnv(insts); pol.rollout(env, greedy=True)
        batch_actions = torch.stack(pol.last_actions)[:, 2]
        env1 = DVRPEnv([insts[2]]); pol.rollout(env1, greedy=True)
        single = torch.stack(pol.last_actions)[:, 0]
    k = len(single)
    assert torch.equal(batch_actions[:k], single)
    assert (batch_actions[k:] == -1).all()


# ---------------------------------------------------------------------------
# B, C, E. Révélation, service, capacité, terminaison (politique aléatoire)
def test_BCE_evenements():
    torch.manual_seed(1)
    pol = DVRPPolicy(information="online").eval()
    insts = generate_dataset(12, base_seed=32_000, n_customers=15)
    env = DVRPEnv(insts)
    env.reset()
    B, N = env.batch_size, env.n_nodes
    prev_vis = env.observe()["visible"].clone()
    served_time = torch.full((B, N), -1.0)
    load = torch.zeros(B)
    with torch.no_grad():
        obs = env.observe(); h = pol.encode(obs); vis = obs["visible"].clone()
        for _ in range(env.max_steps):
            if env.done.all():
                break
            obs = env.observe()
            v = obs["visible"]
            # B : visibilité exacte, monotone
            expected = (env.appear <= env.cur_time.unsqueeze(1)) | (~env.is_dyn)
            expected[:, 0] = True
            assert torch.equal(v, expected)
            assert not (prev_vis & ~v).any()           # jamais retirée
            prev_vis = v.clone()
            # masque : uniquement des requêtes visibles, non servies, compatibles
            m = obs["mask"]
            assert not (m[:, 1:] & ~v[:, 1:]).any()
            assert not (m[:, 1:] & env.served[:, 1:]).any()
            assert not (m[:, 1:] & (env.demand[:, 1:] > env.cur_cap.unsqueeze(1) + 1e-6)).any()
            if (obs["visible"] != vis).any():
                h = pol.encode(obs); vis = obs["visible"].clone()
            a = torch.multinomial(torch.softmax(pol.decode_step(h, obs), -1), 1).squeeze(-1)
            alive = ~env.done
            idx = torch.arange(B)
            # C : jamais deux fois, jamais avant apparition
            is_cust = (a > 0) & alive
            assert not (env.served[idx, a] & is_cust).any()
            t_before = env.cur_time.clone()
            env.step(a)
            for b in torch.nonzero(is_cust).flatten().tolist():
                served_time[b, a[b]] = float(t_before[b])
                assert float(env.appear[b, a[b]]) <= float(t_before[b]) + 1e-6
            # C : capacité
            assert (env.cur_cap >= -1e-6).all()
            load = torch.where(alive & (a == 0), torch.zeros(B), load)
            load = load + torch.where(is_cust, env.demand[idx, a], torch.zeros(B))
            assert (load <= env.capacity + 1e-6).all()
    # E : terminaison
    assert env.done.all()
    assert float(env.n_unserved().sum()) == 0.0
    assert (env.cur_node == 0).all()
    assert (served_time[:, 1:] >= 0).all()          # chaque requête servie une fois


# ---------------------------------------------------------------------------
# D. Attente au dépôt
def test_D_attente_sans_fuite():
    # dépôt + 1 requête statique + 2 requêtes futures
    inst = _make_instance([[50, 50], [60, 50], [10, 10], [90, 90]],
                          [0, 3, 3, 3], [0, 0, 200, 300], [0, 5, 5, 5])
    env = DVRPEnv([inst]); env.reset()
    env.step(torch.tensor([1]))            # servir la requête statique
    env.step(torch.tensor([0]))            # retour au dépôt (t = 25)
    obs = env.observe()
    assert obs["mask"][0].tolist() == [True, False, False, False]
    # l'observation ne contient aucune trace des instants futurs
    assert float(obs["appear"][0, 2:].abs().sum()) == 0.0
    t0 = float(env.cur_time[0])
    env.step(torch.tensor([0]))            # attente
    assert float(env.cur_time[0]) == pytest.approx(200.0)
    assert float(env.total_wait[0]) == pytest.approx(200.0 - t0)
    obs = env.observe()
    assert obs["visible"][0].tolist() == [True, True, True, False]
    assert obs["mask"][0].tolist() == [False, False, True, False]


# ---------------------------------------------------------------------------
# F. Parité
def test_F_parite_inference_rollout():
    torch.manual_seed(2)
    pol = DVRPPolicy(information="online").eval()
    inst = generate_dataset(1, base_seed=33_000, n_customers=15)[0]
    with torch.no_grad():
        env = DVRPEnv([inst]); pol.rollout(env, greedy=True)
    d_roll = float(env.total_distance[0])
    tours, d_inf = solve_instance(pol, inst)
    assert d_inf == pytest.approx(d_roll)
    served = sorted(c for t in tours for c in t)
    assert served == list(range(1, inst.n_total()))


def test_F_regime_incoherent_refuse():
    pol = DVRPPolicy(information="online")
    env = DVRPEnv(generate_dataset(2, base_seed=34_000, n_customers=8), information="clairvoyant")
    with pytest.raises(ValueError):
        pol.rollout(env)


def test_F_entrainement_en_ligne():
    """Une itération d'entraînement en régime en ligne : gradients finis, et
    l'environnement d'entraînement est bien en régime en ligne."""
    from rl.train import train
    pol, hist = train(n_customers=8, epochs=1, batch_size=8, batches_per_epoch=1,
                      device="cpu", seed=0, domain_randomization=False, verbose=False,
                      baseline_eval_size=8, congestion_mode="off", objective="distance",
                      policy_kwargs={"information": "online"})
    assert pol.information == "online"
    assert np.isfinite(hist[0]["loss"])


# ---------------------------------------------------------------------------
# G. Cas limites
def test_G_seul_le_depot_visible():
    inst = _make_instance([[50, 50], [20, 20], [80, 80]], [0, 4, 4], [0, 30, 60], [0, 5, 5])
    torch.manual_seed(3)
    pol = DVRPPolicy(information="online").eval()
    env = DVRPEnv([inst])
    with torch.no_grad():
        pol.rollout(env, greedy=True)
    assert env.done.all() and float(env.n_unserved()[0]) == 0.0
    assert float(env.total_wait[0]) == pytest.approx(30.0)


def test_G_nb_encodages():
    """En ligne, l'encodeur est rappelé à chaque changement du graphe
    observable ; en clairvoyant, une seule fois."""
    inst = _make_instance([[50, 50], [20, 20], [80, 80], [50, 90]],
                          [0, 4, 4, 4], [0, 30, 60, 400], [0, 5, 5, 5])
    with torch.no_grad():
        p = DVRPPolicy(information="online").eval(); p.rollout(DVRPEnv([inst]), greedy=True)
        assert p.n_encodings >= 3
        c = DVRPPolicy(information="clairvoyant").eval()
        c.rollout(DVRPEnv([inst], information="clairvoyant"), greedy=True)
        assert c.n_encodings == 1


# ---------------------------------------------------------------------------
# H. Contrôle : l'ablation clairvoyante exploite l'information cachée
def test_H_clairvoyant_depend_du_futur():
    torch.manual_seed(0)
    pol = DVRPPolicy(information="clairvoyant").eval()
    a, b, k = _pair(900, extra=0)
    with torch.no_grad():
        oa = DVRPEnv([a], information="clairvoyant").observe()
        ob = DVRPEnv([b], information="clairvoyant").observe()
        la = pol.decode_step(pol.encode(oa), oa)[0, :k]
        lb = pol.decode_step(pol.encode(ob), ob)[0, :k]
    valid = la > -1e8
    assert not torch.allclose(la[valid], lb[valid], atol=1e-6)
