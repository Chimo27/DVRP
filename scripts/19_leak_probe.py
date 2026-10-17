"""(Version adaptée à l'API d'observation ; la mesure rapportée pour la
variante antérieure a été faite avec la version précédente du code.)
Sonde de fuite : deux instances avec la même information visible jusqu'à T
(mêmes clients statiques, mêmes clients apparaissant avant T) mais des clients
futurs différents (positions, demandes, instants > T, et NOMBRE différent).
Une politique sans fuite doit prendre exactement les mêmes décisions tant que
le temps courant est < T."""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from data.generator import generate_dataset, generate_instance
from rl.env import DVRPEnv
from rl.policy import DVRPPolicy

def make_pair(seed, n=20, extra=int(os.environ.get("EXTRA", 5))):
    inst = generate_instance(seed=seed, n_customers=n)
    T = 120.0
    rng = np.random.default_rng(seed + 7)
    keep = [0] + [i for i in range(1, n + 1) if inst.appearance_times[i] < T]
    def build(k_hidden):
        idx = list(keep)
        loc = [inst.locations[i] for i in idx]; dem = [inst.demands[i] for i in idx]
        app = [inst.appearance_times[i] for i in idx]; dyn = [inst.is_dynamic[i] for i in idx]
        srv = [inst.service_times[i] for i in idx]
        for _ in range(k_hidden):
            loc.append(rng.uniform(0, 100, 2)); dem.append(float(rng.integers(1, 10)))
            app.append(float(rng.uniform(T + 1, 360))); dyn.append(True); srv.append(float(rng.integers(5, 16)))
        new = copy.deepcopy(inst)
        new.locations = np.array(loc, float); new.demands = np.array(dem, float)
        new.appearance_times = np.array(app, float); new.is_dynamic = np.array(dyn, bool)
        new.service_times = np.array(srv, float)
        d = new.locations[:, None] - new.locations[None]; new.dist_matrix = np.sqrt((d ** 2).sum(-1))
        return new
    nh = n + 1 - len(keep)
    return build(nh), build(nh + extra), T

def trace(policy, inst, T):
    env = DVRPEnv([inst], information=policy.information); env.reset()
    obs = env.observe(); h = policy.encode(obs); vis = obs["visible"].clone()
    acts, logits_list = [], []
    for _ in range(env.max_steps):
        if env.done.all() or float(env.cur_time[0]) >= T: break
        if (obs["visible"] != vis).any():
            h = policy.encode(obs); vis = obs["visible"].clone()
        lg = policy.decode_step(h, obs)
        a = lg.argmax(-1); acts.append(int(a)); logits_list.append(lg[0].clone())
        env.step(a); obs = env.observe()
    return acts, logits_list

if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else "outputs/dynamic/online_s0.pt"
    from rl.train import load_checkpoint
    pol, _ = load_checkpoint(ck); pol.eval()
    diff_actions = diff_logits = 0; maxdev = 0.0; n_dec = 0
    with torch.no_grad():
        for s in range(30):
            a, b, T = make_pair(1000 + s)
            ta, la = trace(pol, a, T); tb, lb = trace(pol, b, T)
            m = min(len(ta), len(tb)); n_dec += m
            if ta[:m] != tb[:m]: diff_actions += 1
            for x, y in zip(la[:m], lb[:m]):
                k = min(len(x), len(y)); vis = (x[:k] > -1e8) & (y[:k] > -1e8)
                dev = float((x[:k][vis] - y[:k][vis]).abs().max()) if vis.any() else 0.0
                maxdev = max(maxdev, dev)
    print(f"{ck}: paires avec décisions différentes avant T = {diff_actions}/30 ; "
          f"écart max des scores visibles = {maxdev:.3e} ; décisions comparées = {n_dec}")
