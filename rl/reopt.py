"""
Heuristique de référence « réoptimisation à chaque événement » (REOPT).

Principe (stratégie classique pour le VRP dynamique, cf. Pillac et al., 2013) :
à chaque fois que l'ensemble des requêtes connues et non servies change
(apparition d'une requête), on résout le problème statique formé de ces
seules requêtes, à partir de la position et de la capacité courantes du
véhicule, puis on suit le plan obtenu jusqu'au prochain événement.

Problème statique résolu (OR-Tools, bibliothèque de routage) :
    - véhicule 0 : part de la position courante avec la capacité restante,
      finit au dépôt (fin de la tournée en cours) ;
    - véhicules 1..K : partent du dépôt avec la capacité pleine (tournées
      suivantes du même véhicule, l'ordre des tournées étant sans effet sur
      la distance) ;
    - objectif : distance totale.
Le plan est construit par « path cheapest arc » puis amélioré par recherche
locale guidée pendant `time_limit` secondes.

L'heuristique n'utilise que l'observation (requêtes apparues) : elle est
myope vis-à-vis des requêtes futures, comme la politique en ligne.
"""
import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from rl.env import DVRPEnv


def _plan(obs, b, time_limit, scale=1000):
    vis = obs["visible"][b].cpu().numpy()
    served = obs["served"][b].cpu().numpy()
    pending = [i for i in range(1, len(vis)) if vis[i] and not served[i]]
    if not pending:
        return []
    cur = int(obs["cur_node"][b])
    cap_left = int(round(float(obs["cur_cap"][b])))
    Q = int(round(float(obs["capacity"][b])))
    dem = obs["demand"][b].cpu().numpy()
    D = obs["dist"][b].cpu().numpy()

    # nœuds du sous-problème : 0 = dépôt, 1 = position courante, 2.. = requêtes
    nodes = [0, cur] + pending
    n = len(nodes)
    dm = np.array([[D[a, c] for c in nodes] for a in nodes]) * scale
    q = [0, 0] + [int(round(dem[i])) for i in pending]
    K = int(np.ceil(sum(q) / Q)) + 1
    starts = [1] + [0] * K
    ends = [0] * (K + 1)
    caps = [cap_left] + [Q] * K

    manager = pywrapcp.RoutingIndexManager(n, K + 1, starts, ends)
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(
        lambda i, j: int(dm[manager.IndexToNode(i), manager.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    dcb = routing.RegisterUnaryTransitCallback(lambda i: q[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(dcb, 0, caps, True, "cap")
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit * 1000))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    seq = []
    for v in range(K + 1):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node >= 2:
                route.append(nodes[node])
            idx = sol.Value(routing.NextVar(idx))
        if v == 0:
            seq += route + [0]
        elif route:
            seq += route + [0]
    return seq


@torch.no_grad()
def reoptimization_policy(env, time_limit=0.2):
    """Déroule REOPT sur toutes les instances du lot (séquentiellement)."""
    env.reset()
    B = env.batch_size
    plans = [None] * B
    known = [None] * B
    n_solves = 0
    for _ in range(env.max_steps):
        if env.done.all():
            break
        obs = env.observe()
        actions = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            if bool(env.done[b]):
                continue
            vis_set = tuple(np.nonzero(obs["visible"][b].cpu().numpy())[0])
            if plans[b] is None or vis_set != known[b] or not plans[b]:
                p = _plan(obs, b, time_limit)
                n_solves += 1
                plans[b] = list(p) if p else []
                known[b] = vis_set
                # au dépôt, le plan commence par la tournée suivante
                if plans[b] and int(obs["cur_node"][b]) == 0 and plans[b][0] == 0:
                    plans[b].pop(0)
            if plans[b]:
                a = plans[b].pop(0)
                if a == 0 and int(obs["cur_node"][b]) == 0 and bool(obs["mask"][b, 1:].any()):
                    a = plans[b].pop(0) if plans[b] else 0
            else:
                a = 0                                  # retour / attente au dépôt
            if not bool(obs["mask"][b, a]):
                raise RuntimeError("REOPT : action non admissible")
            actions[b] = a
        env.step(actions)
    m = env.metrics()
    m["n_solves"] = n_solves
    return env.final_reward(), m
