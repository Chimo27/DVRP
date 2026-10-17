"""Contrôle indépendant de CP-SAT : optimum exact du CVRP statique (flotte non
bornée, distance) par programmation dynamique (Held-Karp sur chaque sous-ensemble
réalisable, puis partition optimale de l'ensemble des clients en tournées)."""
import sys, json
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from data.generator import generate_dataset

def cvrp_dp(D, q, Q):
    n = len(q) - 1                       # clients 1..n
    full = 1 << n
    INF = float("inf")
    # dp[S][j] : chemin dépôt -> ... -> j couvrant S (j dans S)
    dp = np.full((full, n), INF)
    for j in range(n):
        dp[1 << j][j] = D[0][j + 1]
    load = np.zeros(full)
    for S in range(1, full):
        lb = (S & -S).bit_length() - 1
        load[S] = load[S ^ (1 << lb)] + q[lb + 1]
    for S in range(1, full):
        if load[S] > Q: continue
        row = dp[S]
        for j in range(n):
            v = row[j]
            if v == INF: continue
            for k in range(n):
                if S >> k & 1: continue
                T = S | (1 << k)
                if load[T] > Q: continue
                c = v + D[j + 1][k + 1]
                if c < dp[T][k]: dp[T][k] = c
    tour = np.full(full, INF)
    for S in range(1, full):
        if load[S] <= Q:
            tour[S] = min(dp[S][j] + D[j + 1][0] for j in range(n) if S >> j & 1)
    f = np.full(full, INF); f[0] = 0.0
    for S in range(1, full):
        lb = S & -S
        rest = S ^ lb
        T = rest
        best = INF
        while True:
            sub = T | lb
            if tour[sub] < INF:
                c = tour[sub] + f[S ^ sub]
                if c < best: best = c
            if T == 0: break
            T = (T - 1) & rest
        f[S] = best
    return f[full - 1]

cp = json.load(open("outputs/final/exact_cpsat.json"))
out = {}
for n in (8, 10, 12):
    insts = generate_dataset(30, base_seed=20_000_000 + n, n_customers=n)
    rows = []
    for k, inst in enumerate(insts):
        D = (inst.dist_matrix * 1000).astype(int).astype(float)   # même arrondi que CP-SAT
        v = cvrp_dp(D, inst.demands.astype(int), int(inst.capacity)) / 1000
        rows.append(dict(instance=k, dp=v, cpsat=cp[str(n)][k]["cpsat"]))
    out[n] = rows
    print(n, "max|dp-cpsat| =", max(abs(r["dp"] - r["cpsat"]) for r in rows), flush=True)
json.dump(out, open("outputs/final/exact_dp.json", "w"), indent=1)
