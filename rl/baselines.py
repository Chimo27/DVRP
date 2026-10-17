"""
Heuristiques de référence.

Sans point de comparaison, un score de -400 ne veut rien dire. Ces
heuristiques tournent dans le MÊME environnement que l'agent RL, avec le
même masquage, donc les résultats sont directement comparables.

    random_policy          : borne haute (pire cas raisonnable)
    nearest_neighbor_policy: heuristique gloutonne classique — la vraie
                              référence à battre pour le modèle RL
"""

import torch


@torch.no_grad()
def random_policy(env):
    """Choisit uniformément parmi les actions valides."""
    state = env.reset()
    for _ in range(env.max_steps):
        if env.done.all():
            break
        m = env.observe()["mask"].float()
        probs = m / m.sum(dim=1, keepdim=True).clamp(min=1e-9)
        action = torch.multinomial(probs, 1).squeeze(-1)
        state, _ = env.step(action)
    return env.final_reward(), env.metrics()


@torch.no_grad()
def nearest_neighbor_policy(env):
    """
    Va toujours au client valide le plus proche ; retourne au dépôt quand
    aucun client n'est servable (recharge ou attente d'une arrivée).
    """
    state = env.reset()
    B = env.batch_size
    idx = torch.arange(B, device=env.device)

    for _ in range(env.max_steps):
        if env.done.all():
            break

        obs = env.observe()                                   # information disponible
        mask = obs["mask"]
        d_from_cur = obs["dist"][idx, env.cur_node]           # (B, N)

        # Distance infinie sur les actions invalides -> jamais choisies
        d_masked = torch.where(mask, d_from_cur, torch.full_like(d_from_cur, float("inf")))

        # On écarte le dépôt du choix "plus proche" tant qu'un client est
        # disponible : sinon le véhicule y reviendrait en permanence, le
        # dépôt étant souvent le nœud le plus proche.
        customer_available = mask[:, 1:].any(dim=1)
        d_choice = d_masked.clone()
        d_choice[:, 0] = torch.where(
            customer_available,
            torch.full((B,), float("inf"), device=env.device),
            d_masked[:, 0],
        )

        action = d_choice.argmin(dim=1)
        state, _ = env.step(action)

    return env.final_reward(), env.metrics()


BASELINES = {
    "random": random_policy,
    "nearest_neighbor": nearest_neighbor_policy,
}
