"""
Environnement DVRP vectorisé (batché) pour l'entraînement RL.

MODÈLE DE DÉCISION
------------------
Véhicule unique à recharges multiples (convention standard de la
littérature RL-VRP) : le véhicule retourne au dépôt pour recharger sa
capacité, ce qui simule une flotte de plusieurs tournées successives.

ÉTAT
    position courante, temps courant, capacité restante,
    clients servis, clients révélés (r_i <= t)

ACTION
    choisir le prochain nœud (client ou dépôt)

MASQUAGE (garantit la validité de toute action échantillonnée)
    client valide  <=> non servi ET révélé ET demande <= capacité restante
    dépôt valide   <=> soit on n'y est pas (retour/recharge),
                       soit on y est mais aucun client n'est servable
                       (= attente explicite d'une future arrivée)

TERMINAISON
    tous les clients servis ET véhicule au dépôt.

    La terminaison est GARANTIE : la validation impose demande_i <= capacité,
    donc après une recharge au dépôt tout client révélé devient servable, et
    l'attente au dépôt fait avancer le temps jusqu'à la prochaine révélation.
    Aucun blocage n'est possible.

RÉCOMPENSE
    -(distance totale). Objectif propre et non contournable : l'épisode ne
    peut se terminer qu'une fois TOUS les clients servis, donc le modèle ne
    peut pas "gagner" en ignorant des clients.

    (Une pénalité `unserved_penalty` existe uniquement comme filet de
    sécurité si le garde-fou max_steps coupe l'épisode — ce qui ne devrait
    jamais arriver en fonctionnement normal.)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ENV_CONFIG
from data.travel_time import compute_travel_time, env_time_to_clock


class DVRPEnv:
    def __init__(self, instances, device="cpu",
                 unserved_penalty=None, max_steps_factor=None,
                 congestion_mode=None, congestion_params=None, seed=None,
                 objective=None, time_weight=None, information="online"):
        self.device = device
        # Régime d'information de la POLITIQUE (le simulateur, lui, connaît
        # toujours l'instance complète) :
        #   "online"      : la politique n'observe que les requêtes déjà
        #                   apparues (r_i <= t) — problème dynamique.
        #   "clairvoyant" : la politique observe toutes les requêtes dès t=0
        #                   (ablation, comportement de la campagne initiale).
        if information not in ("online", "clairvoyant"):
            raise ValueError("information doit être 'online' ou 'clairvoyant'")
        self.information = information
        self.unserved_penalty = (unserved_penalty if unserved_penalty is not None
                                  else ENV_CONFIG["unserved_penalty"])
        self.max_steps_factor = (max_steps_factor if max_steps_factor is not None
                                  else ENV_CONFIG["max_steps_factor"])

        # --- Objectif d'optimisation (cf. final_reward) ---
        self.objective = objective or ENV_CONFIG.get("objective", "distance")
        self.time_weight = (time_weight if time_weight is not None
                             else ENV_CONFIG.get("time_weight", 1.0))

        # --- Congestion (cf. data/travel_time.py) ---
        self.congestion_mode = (congestion_mode if congestion_mode is not None
                                 else ENV_CONFIG.get("congestion_mode", "off"))
        self.congestion_params = congestion_params or ENV_CONFIG.get("congestion_params")

        # Générateur dédié au bruit de congestion : sans lui, deux rollouts
        # sur la même instance donneraient des temps différents, ce qui
        # fausserait la comparaison politique / baseline dans REINFORCE.
        self.generator = torch.Generator(device=device)
        self._seed = seed if seed is not None else 0
        self.generator.manual_seed(self._seed)

        self.batch_size = len(instances)
        self.n_nodes = instances[0].n_total()

        def stack(attr):
            return np.array([getattr(i, attr) for i in instances])

        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)

        self.loc = t(stack("locations"))
        self.demand = t(stack("demands"))
        self.appear = t(stack("appearance_times"))
        self.is_dyn = t(stack("is_dynamic"), torch.bool)
        self.service = t(stack("service_times"))
        self.dist = t(stack("dist_matrix"))
        self.capacity = t(np.array([i.capacity for i in instances]))

        self.velocity = float(instances[0].metadata.get("velocity", 1.0))
        self.horizon = float(instances[0].metadata.get("planning_horizon", (0, 480))[1])

        self.max_steps = self.n_nodes * self.max_steps_factor
        self.reset()

    # -----------------------------------------------------------------
    def reset(self):
        B, N = self.batch_size, self.n_nodes
        dev = self.device

        # Rejoue la même séquence de bruit de congestion à chaque épisode :
        # indispensable pour que le rollout échantillonné et le rollout de
        # la baseline soient comparés dans des conditions identiques.
        self.generator.manual_seed(self._seed)

        self.served = torch.zeros(B, N, dtype=torch.bool, device=dev)
        self.served[:, 0] = True                      # le dépôt n'est pas "à servir"
        self.cur_node = torch.zeros(B, dtype=torch.long, device=dev)
        self.cur_time = torch.zeros(B, dtype=torch.float32, device=dev)
        self.cur_cap = self.capacity.clone()
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

        self.total_distance = torch.zeros(B, dtype=torch.float32, device=dev)
        self.total_wait = torch.zeros(B, dtype=torch.float32, device=dev)
        self.n_steps = 0
        return self.get_state()

    # -----------------------------------------------------------------
    def _revealed(self):
        """Un client statique est toujours connu ; un dynamique seulement si r_i <= t."""
        return (~self.is_dyn) | (self.appear <= self.cur_time.unsqueeze(1))

    def get_action_mask(self):
        revealed = self._revealed()
        feasible_cap = self.demand <= self.cur_cap.unsqueeze(1) + 1e-6

        customer_mask = (~self.served) & revealed & feasible_cap
        customer_mask[:, 0] = False

        at_depot = self.cur_node == 0
        any_customer = customer_mask.any(dim=1)
        # Dépôt valide si :
        #   - on n'y est pas (retour / recharge), OU
        #   - on y est mais aucun client n'est servable (attente), OU
        #   - tout est servi (action terminale / no-op), OU
        #   - l'épisode est déjà fini (évite une ligne de masque toute
        #     à False, qui produirait un NaN dans le softmax du batch)
        # (« tout est servi » implique « aucun client servable » : le masque
        # ne dépend donc que des requêtes apparues.)
        depot_ok = (~at_depot) | (~any_customer) | self.done

        mask = customer_mask.clone()
        mask[:, 0] = depot_ok
        return mask

    def get_state(self):
        return {
            "loc": self.loc, "demand": self.demand, "dist": self.dist,
            "appear": self.appear, "is_dyn": self.is_dyn, "service": self.service,
            "cur_node": self.cur_node, "cur_time": self.cur_time,
            "cur_cap": self.cur_cap, "capacity": self.capacity,
            "served": self.served, "revealed": self._revealed(),
            "mask": self.get_action_mask(), "done": self.done,
            "horizon": self.horizon,
        }

    def observe(self):
        """
        Observation transmise à la politique : SEULE porte d'entrée de
        l'information vers le modèle.

        En régime "online", une requête non encore apparue est absente de
        l'observation : ses attributs (position, demande, durée de service,
        instant d'apparition, statut dynamique) et ses distances sont mis à
        zéro, elle est exclue de la visibilité (donc du passage de messages,
        de l'attention, du résumé du graphe et des comptages), et elle n'est
        jamais sélectionnable. Les comptages exposés portent uniquement sur
        les requêtes visibles : le nombre total de requêtes de l'instance
        n'est pas observable.
        """
        if self.information == "online":
            vis = self._revealed().clone()
        else:
            vis = torch.ones_like(self.served)
        vis[:, 0] = True                                   # le dépôt est toujours connu
        vf = vis.float()
        served_vis = self.served & vis
        return {
            "visible": vis,
            "loc": self.loc * vf.unsqueeze(-1),
            "demand": self.demand * vf,
            "service": self.service * vf,
            "appear": self.appear * vf,
            "is_dyn": self.is_dyn & vis,
            "dist": self.dist * vf.unsqueeze(2) * vf.unsqueeze(1),
            "served": served_vis,
            "n_visible": vis[:, 1:].float().sum(1),
            "n_served_visible": served_vis[:, 1:].float().sum(1),
            "cur_node": self.cur_node, "cur_time": self.cur_time,
            "cur_cap": self.cur_cap, "capacity": self.capacity,
            "horizon": self.horizon,
            "mask": self.get_action_mask(),
        }

    # -----------------------------------------------------------------
    def step(self, action):
        B = self.batch_size
        idx = torch.arange(B, device=self.device)
        active = ~self.done

        travel_dist = self.dist[idx, self.cur_node, action]

        # --- Temps de trajet dépendant de l'heure (congestion) ---
        # Le temps de trajet dépend de l'instant de DÉPART : partir à 8h
        # (heure de pointe) coûte plus cher que partir à midi. cf.
        # data/travel_time.py pour les modes disponibles.
        if self.congestion_mode == "off":
            travel_time = travel_dist / self.velocity
        else:
            clock = env_time_to_clock(self.cur_time, self.congestion_params)
            travel_time = compute_travel_time(
                travel_dist, clock, params=self.congestion_params,
                mode=self.congestion_mode, velocity=self.velocity,
                generator=self.generator,
            )

        at_depot_before = self.cur_node == 0
        waiting = (action == 0) & at_depot_before      # attente explicite sur place

        # --- Distance : on ne compte pas une attente sur place ---
        self.total_distance = self.total_distance + travel_dist * (active & ~waiting).float()

        # --- Avancée du temps ---
        new_time = self.cur_time + travel_time + self.service[idx, action]

        # Attente : saute jusqu'à la prochaine révélation connue.
        # Sans ce mécanisme, choisir le dépôt en y étant déjà ne ferait pas
        # avancer le temps et l'épisode se bloquerait (aucun client dynamique
        # ne serait jamais révélé).
        pending = self.is_dyn & (~self.served) & (self.appear > self.cur_time.unsqueeze(1))
        next_reveal = torch.where(
            pending, self.appear, torch.full_like(self.appear, float("inf"))
        ).min(dim=1).values
        has_pending = torch.isfinite(next_reveal)
        wait_target = torch.where(has_pending, next_reveal, self.cur_time)
        waited_time = torch.clamp(wait_target - self.cur_time, min=0.0)
        self.total_wait = self.total_wait + waited_time * (active & waiting).float()
        new_time = torch.where(waiting, self.cur_time + waited_time, new_time)

        # --- Capacité : rechargée au dépôt, sinon consommée ---
        new_cap = torch.where(
            action == 0, self.capacity, self.cur_cap - self.demand[idx, action]
        )

        self.cur_time = torch.where(active, new_time, self.cur_time)
        self.cur_cap = torch.where(active, new_cap, self.cur_cap)
        self.cur_node = torch.where(active, action, self.cur_node)
        self.served[idx, action] = self.served[idx, action] | active

        all_served = ~(~self.served[:, 1:]).any(dim=1)
        self.done = self.done | (all_served & (self.cur_node == 0))

        self.n_steps += 1
        return self.get_state(), self.done.clone()

    # -----------------------------------------------------------------
    def n_unserved(self):
        return (~self.served[:, 1:]).float().sum(dim=1)

    def final_reward(self):
        """
        Récompense à MAXIMISER, selon l'objectif choisi :

          "distance" : -(distance totale)          [défaut historique]
          "time"     : -(makespan)                  = durée totale de service
          "combined" : -(distance + time_weight * makespan)

        POURQUOI CE CHOIX EST CRITIQUE AVEC LA CONGESTION
        --------------------------------------------------
        La congestion allonge le TEMPS de trajet, pas la distance parcourue.
        Avec objective="distance", le modèle subit donc les heures de pointe
        sans aucune incitation à les éviter : la congestion est présente dans
        la simulation mais absente de l'objectif d'apprentissage.

        Pour que le modèle apprenne réellement à composer avec la congestion
        (partir plus tôt, réordonner les visites, attendre au dépôt), il faut
        que le temps entre dans la récompense -> "time" ou "combined".

        Dans tous les cas, la pénalité pour clients non servis reste active
        comme filet de sécurité (ne devrait jamais se déclencher).
        """
        penalty = self.n_unserved() * self.unserved_penalty

        if self.objective == "distance":
            cost = self.total_distance
        elif self.objective == "time":
            cost = self.cur_time
        elif self.objective == "combined":
            cost = self.total_distance + self.time_weight * self.cur_time
        else:
            raise ValueError("objective doit être 'distance', 'time' ou 'combined'")

        return -(cost + penalty)

    def metrics(self):
        """Métriques séparées, pour un diagnostic lisible de l'entraînement."""
        return {
            "distance": self.total_distance.detach(),
            "makespan": self.cur_time.detach(),
            "wait": self.total_wait.detach(),
            "unserved": self.n_unserved().detach(),
            "completed": self.done.float().detach(),
        }
