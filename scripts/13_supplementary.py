"""Campagne complémentaire (révision du rapport, CPU, budget réduit).

Quatre variantes, trois graines chacune, même configuration d'instances que le
run « sans congestion » officiel (20 clients, DoD 0,5, objectif temps,
congestion désactivée), budget réduit : 100 époques x 10 lots x 64 instances.
  A  ref       : clairvoyant (encodeur voit les clients futurs), arêtes, baseline figée
  B  masque    : non clairvoyant (clients non révélés masqués, ré-encodage)
  C  sans_aretes : comme A, sans attribut d'arête dans GATv2
  D  mobile    : comme A, baseline gloutonne de la politique courante
Évaluation : 256 instances de test (graines 2e7), par instance, vs plus proche
voisin ; balayage du DoD pour A et B. Écrit outputs/supp/resultats_supp.json.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
torch.set_num_threads(2)
from rl.train import train, save_checkpoint
from rl.env import DVRPEnv
from rl.baselines import nearest_neighbor_policy
from data.generator import generate_dataset

OUT = "outputs/supp"; os.makedirs(OUT, exist_ok=True)
PATH = f"{OUT}/resultats_supp.json"
EPOCHS, BPE, BS = int(os.environ.get("EP", 100)), 10, 64
VARIANTS = {
    "A_ref":         dict(policy_kwargs={"information": "clairvoyant"}, baseline_mode="frozen"),
    "B_masque":      dict(policy_kwargs={"observe_future": False}, baseline_mode="frozen"),
    "C_sans_aretes": dict(policy_kwargs={"information": "clairvoyant", "use_edge_attr": False}, baseline_mode="frozen"),
    "D_mobile":      dict(policy_kwargs={"information": "clairvoyant"}, baseline_mode="moving"),
}
ENVKW = dict(congestion_mode="off", objective="time")

def per_instance(policy, n=20, dod=0.5, k=256):
    insts = generate_dataset(k, base_seed=20_000_000, n_customers=n, degree_of_dynamism=dod)
    _, mn = nearest_neighbor_policy(DVRPEnv(insts, **ENVKW, seed=1))
    policy.eval()
    with torch.no_grad():
        _, _, _, mr = policy.rollout(DVRPEnv(insts, **ENVKW, seed=1, information=policy.information), greedy=True)
    f = lambda t: [round(float(v), 3) for v in t]
    return {"rl_dist": f(mr["distance"]), "nn_dist": f(mn["distance"]),
            "rl_mk": f(mr["makespan"]), "nn_mk": f(mn["makespan"]),
            "rl_wait": f(mr["wait"]), "nn_wait": f(mn["wait"]),
            "dod_real": [float(i.is_dynamic[1:].mean()) for i in insts]}

res = json.load(open(PATH)) if os.path.exists(PATH) else {}
res["config"] = dict(epochs=EPOCHS, batches_per_epoch=BPE, batch_size=BS, n_customers=20,
                     degree_of_dynamism=0.5, **ENVKW, device="cpu", torch=torch.__version__)
order = [(v, s) for s in (0, 1, 2) for v in VARIANTS]
for name, seed in order:
    key = f"{name}_s{seed}"
    if key in res: continue
    t0 = time.time()
    pol, hist = train(n_customers=20, epochs=EPOCHS, batch_size=BS, batches_per_epoch=BPE,
                      device="cpu", seed=seed, domain_randomization=False, log_every=25,
                      verbose=True, **ENVKW, **VARIANTS[name])
    rec = {"variant": name, "seed": seed, "train_s": time.time() - t0,
           "baseline_updates": int(sum(h["baseline_updated"] for h in hist)),
           "history": [{k: h[k] for k in ("epoch", "eval_distance", "train_distance", "gap_vs_nn_pct")} for h in hist],
           "test": per_instance(pol)}
    if name in ("A_ref", "B_masque"):
        rec["dod"] = {str(d): per_instance(pol, dod=d) for d in (0.0, 1.0)}
    save_checkpoint(pol, hist, f"{OUT}/{key}.pt")
    res[key] = rec
    json.dump(res, open(PATH, "w"))
    t = rec["test"]; g = (np.mean(t["rl_dist"]) - np.mean(t["nn_dist"])) / np.mean(t["nn_dist"]) * 100
    print(f"== {key}: RL {np.mean(t['rl_dist']):.1f} NN {np.mean(t['nn_dist']):.1f} ({g:+.1f}%) "
          f"[{rec['train_s']/60:.1f} min]", flush=True)
print("TERMINE", flush=True)
