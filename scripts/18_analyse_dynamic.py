"""Analyse de la campagne dynamique (outputs/dynamic/resultats.json).

Pour chaque régime (en ligne, clairvoyant) : moyenne et écart-type entre
graines, écarts au PPV et à REOPT, tests appariés (Student, Wilcoxon) sur les
instances, nombre d'instances perdues ; balayage du DoD ; tailles ; écart à
l'optimum statique (CP-SAT). Écrit outputs/dynamic/analyse.json."""
import json
import os
import sys

import numpy as np
from scipy import stats

R = json.load(open("outputs/dynamic/resultats.json"))
OPT = json.load(open("outputs/final/exact_cpsat.json"))
B = R["baselines"]
seeds = sorted({int(k.split("_s")[1]) for k in R if k.startswith(("online_s", "clairvoyant_s"))})


def gap(a, b):
    return (np.mean(a) - np.mean(b)) / np.mean(b) * 100


def runs(info):
    return [R[f"{info}_s{s}"] for s in seeds if f"{info}_s{s}" in R]


out = {"seeds": seeds, "config": R["config"]}
for name in B:
    b = B[name]
    out.setdefault("baselines", {})[name] = {
        "nn": float(np.mean(b["nn"]["dist"])), "reopt": float(np.mean(b["reopt"]["dist"])),
        "random": float(np.mean(b["random"]["dist"])),
        "reopt_vs_nn": gap(b["reopt"]["dist"], b["nn"]["dist"]),
        "p_reopt_nn": float(stats.ttest_rel(b["reopt"]["dist"], b["nn"]["dist"]).pvalue),
        "wait_nn": float(np.mean(b["nn"]["wait"])), "wait_reopt": float(np.mean(b["reopt"]["wait"])),
        "mk_nn": float(np.mean(b["nn"]["mk"])), "mk_reopt": float(np.mean(b["reopt"]["mk"])),
        "reopt_solves_per_inst": b["reopt_solves"] / len(b["nn"]["dist"]),
        "dod_real": float(np.mean(b["dod_real"]))}

for info in ("online", "clairvoyant"):
    rs = runs(info)
    if not rs:
        continue
    d = {}
    for name in B:
        nn, ro = B[name]["nn"]["dist"], B[name]["reopt"]["dist"]
        per = [r["eval"][name]["dist"] for r in rs]
        mean_inst = np.mean(per, axis=0)            # moyenne des graines, instance par instance
        e = {"dist": [float(np.mean(x)) for x in per],
             "gap_nn": [gap(x, nn) for x in per], "gap_reopt": [gap(x, ro) for x in per],
             "p_nn": [float(stats.ttest_rel(x, nn).pvalue) for x in per],
             "p_reopt": [float(stats.ttest_rel(x, ro).pvalue) for x in per],
             "worse_than_nn": [int(np.sum(np.array(x) > np.array(nn) + 1e-6)) for x in per],
             "worse_than_reopt": [int(np.sum(np.array(x) > np.array(ro) + 1e-6)) for x in per],
             "wait": [float(np.mean(r["eval"][name]["wait"])) for r in rs],
             "mk": [float(np.mean(r["eval"][name]["mk"])) for r in rs],
             "n_enc": [float(r["eval"][name]["n_encodings"]) for r in rs],
             "pooled_gap_nn": gap(mean_inst, nn), "pooled_gap_reopt": gap(mean_inst, ro),
             "pooled_p_nn": float(stats.ttest_rel(mean_inst, nn).pvalue),
             "pooled_p_reopt": float(stats.ttest_rel(mean_inst, ro).pvalue),
             "pooled_wilcoxon_reopt": float(stats.wilcoxon(mean_inst, ro).pvalue)}
        for k in ("dist", "gap_nn", "gap_reopt"):
            e[k + "_mean"] = float(np.mean(e[k]))
            e[k + "_sd"] = float(np.std(e[k], ddof=1)) if len(e[k]) > 1 else None
        if name.startswith("exact"):
            n = name[5:]
            opt = np.array([row["cpsat"] for row in OPT[n]])
            e["gap_opt"] = [float(np.mean((np.array(x) - opt) / opt * 100)) for x in per]
            e["gap_opt_nn"] = float(np.mean((np.array(nn) - opt) / opt * 100))
            e["gap_opt_reopt"] = float(np.mean((np.array(ro) - opt) / opt * 100))
        d[name] = e
    out[info] = {"runs": len(rs), "train_min": [r["train_s"] / 60 for r in rs],
                 "updates": [r["baseline_updates"] for r in rs], "eval": d}

# valeur de l'information : en ligne vs clairvoyant, appariée par graine et par instance
if runs("online") and runs("clairvoyant"):
    vi = {}
    for name in B:
        rows = []
        for s in seeds:
            a, c = R.get(f"online_s{s}"), R.get(f"clairvoyant_s{s}")
            if a and c:
                x, y = np.array(a["eval"][name]["dist"]), np.array(c["eval"][name]["dist"])
                rows.append({"seed": s, "diff_pct": float((x.mean() - y.mean()) / y.mean() * 100),
                             "p": float(stats.ttest_rel(x, y).pvalue)})
        vi[name] = rows
    out["online_vs_clairvoyant"] = vi

json.dump(out, open("outputs/dynamic/analyse.json", "w"), indent=1)


def show(x):
    return np.round(x, 2).tolist() if isinstance(x, list) else (round(x, 3) if isinstance(x, float) else x)


print("RÉFÉRENCES")
for k, v in out["baselines"].items():
    print(" ", k, {kk: show(vv) for kk, vv in v.items()})
for info in ("online", "clairvoyant"):
    if info in out:
        print(info.upper(), "runs", out[info]["runs"], "train min", show(out[info]["train_min"]))
        for k, v in out[info]["eval"].items():
            print(" ", k, {kk: show(vv) for kk, vv in v.items() if kk in (
                "dist_mean", "dist_sd", "gap_nn", "gap_reopt", "p_reopt", "worse_than_reopt",
                "pooled_gap_reopt", "pooled_p_reopt", "wait", "n_enc", "gap_opt", "gap_opt_nn",
                "gap_opt_reopt", "worse_than_nn")})
if "online_vs_clairvoyant" in out:
    print("EN LIGNE vs CLAIRVOYANT")
    for k, v in out["online_vs_clairvoyant"].items():
        print(" ", k, [(r["seed"], round(r["diff_pct"], 2), float(f"{r['p']:.2g}")) for r in v])
