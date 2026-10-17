"""Vérification exacte (CP-SAT) des valeurs « lower_bound » du comparatif OR-Tools.
Régénère les 90 instances du script 08 (graines 20_000_000 + n), résout le CVRP
statique à l'optimum prouvé avec CP-SAT, compare avec la valeur GLS stockée."""
import sys, json, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import numpy as np, torch
from ortools.sat.python import cp_model
from data.generator import generate_dataset
from rl.env import DVRPEnv
from rl.baselines import nearest_neighbor_policy

TEST = 20_000_000
stored = json.load(open("outputs/final/comparaison_exacte.json"))["results"]

def solve(inst, scale=1000, tl=120):
    n = inst.n_total(); D = (inst.dist_matrix * scale).astype(int)
    q = inst.demands.astype(int); Q = int(inst.capacity)
    m = cp_model.CpModel(); arcs = []; x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = m.NewBoolVar(""); arcs.append((i, j, x[i, j]))
    m.AddMultipleCircuit(arcs)
    u = [m.NewIntVar(0, Q, "") for _ in range(n)]
    for i in range(1, n):
        m.Add(u[i] >= int(q[i]))
        m.Add(u[i] == int(q[i])).OnlyEnforceIf(x[0, i])
        for j in range(1, n):
            if i != j:
                m.Add(u[j] == u[i] + int(q[j])).OnlyEnforceIf(x[i, j])
    m.Minimize(sum(int(D[i, j]) * v for (i, j), v in x.items()))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 2
    st = s.Solve(m)
    return s.ObjectiveValue() / scale, st == cp_model.OPTIMAL, s.WallTime()

out = {}
for n in (8, 10, 12):
    insts = generate_dataset(30, base_seed=TEST + n, n_customers=n)
    _, mnn = nearest_neighbor_policy(DVRPEnv(insts))
    rows = []
    for k, inst in enumerate(insts):
        val, opt, wt = solve(inst)
        s = stored[str(n)]["rows"][k]
        rows.append(dict(instance=k, cpsat=val, proven=opt, wall=wt, gls=s["lower_bound"],
                         nn_regen=float(mnn["distance"][k]), nn_stored=s["nn_distance"]))
    out[n] = rows
    print(n, "proven", sum(r["proven"] for r in rows), "max|gls-cpsat|",
          max(abs(r["gls"] - r["cpsat"]) for r in rows),
          "max|nn diff|", max(abs(r["nn_regen"] - r["nn_stored"]) for r in rows),
          "mean wall", np.mean([r["wall"] for r in rows]), flush=True)
json.dump(out, open("outputs/final/exact_cpsat.json", "w"), indent=1)
