"""Analyse de la campagne complémentaire : moyennes, dispersion entre graines,
tests appariés (Student et Wilcoxon) par instance. Écrit outputs/supp/analyse.json."""
import json, sys, numpy as np
from scipy import stats
R = json.load(open("outputs/supp/resultats_supp.json"))
out = {}
def gap(a, b): return (np.mean(a) - np.mean(b)) / np.mean(b) * 100
variants = ["A_ref", "B_masque", "C_sans_aretes", "D_mobile"]
for v in variants:
    runs = [R[f"{v}_s{s}"] for s in range(3) if f"{v}_s{s}" in R]
    if not runs: continue
    g = [gap(r["test"]["rl_dist"], r["test"]["nn_dist"]) for r in runs]
    d = [float(np.mean(r["test"]["rl_dist"])) for r in runs]
    p = [float(stats.ttest_rel(r["test"]["rl_dist"], r["test"]["nn_dist"]).pvalue) for r in runs]
    wins = [int(np.sum(np.array(r["test"]["rl_dist"]) > np.array(r["test"]["nn_dist"]))) for r in runs]
    wait = [float(np.mean(r["test"]["rl_wait"])) for r in runs]
    out[v] = dict(n=len(runs), dist=d, gap=g, gap_mean=float(np.mean(g)), gap_sd=float(np.std(g, ddof=1)) if len(g) > 1 else None,
                  p_vs_nn=p, rl_worse_than_nn=wins, wait_mean=wait,
                  nn_dist=float(np.mean(runs[0]["test"]["nn_dist"])), nn_wait=float(np.mean(runs[0]["test"]["nn_wait"])),
                  train_min=[r["train_s"] / 60 for r in runs], updates=[r["baseline_updates"] for r in runs],
                  mk=[float(np.mean(r["test"]["rl_mk"])) for r in runs])
# comparaisons appariées entre variantes, graine par graine (mêmes instances de test)
comp = {}
for v in variants[1:]:
    rows = []
    for s in range(3):
        a, b = R.get(f"A_ref_s{s}"), R.get(f"{v}_s{s}")
        if a and b:
            x, y = np.array(b["test"]["rl_dist"]), np.array(a["test"]["rl_dist"])
            rows.append(dict(seed=s, diff_pct=float((y.mean() and (x.mean() - y.mean()) / y.mean() * 100)),
                             p_t=float(stats.ttest_rel(x, y).pvalue), p_w=float(stats.wilcoxon(x, y).pvalue)))
    comp[f"{v}_vs_A_ref"] = rows
out["comparaisons"] = comp
# balayage DoD
dod = {}
for v in ("A_ref", "B_masque"):
    for d in ("0.0", "1.0"):
        rs = [R[f"{v}_s{s}"]["dod"][d] for s in range(3) if f"{v}_s{s}" in R]
        if rs:
            dod[f"{v}_dod{d}"] = dict(rl=[float(np.mean(r["rl_dist"])) for r in rs],
                                     gap=[gap(r["rl_dist"], r["nn_dist"]) for r in rs],
                                     nn=float(np.mean(rs[0]["nn_dist"])),
                                     wait_rl=[float(np.mean(r["rl_wait"])) for r in rs],
                                     wait_nn=float(np.mean(rs[0]["nn_wait"])))
out["dod"] = dod
json.dump(out, open("outputs/supp/analyse.json", "w"), indent=1)
for k, v in out.items():
    if k in ("comparaisons", "dod"):
        print(k); [print("  ", kk, vv) for kk, vv in v.items()]
    else:
        print(k, {kk: (np.round(vv, 3).tolist() if isinstance(vv, list) else (round(vv, 3) if isinstance(vv, float) else vv)) for kk, vv in v.items()})
