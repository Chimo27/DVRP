"""
Configuration centralisée du projet.

Toutes les valeurs par défaut sont ici, pas dispersées dans le code.
Elles restent surchargeables à l'appel de chaque fonction.
"""

# ---------------------------------------------------------------------------
# Génération d'instances
# ---------------------------------------------------------------------------

INSTANCE_CONFIG = {
    "n_customers": 20,
    "coord_range": (0, 100),
    "depot_position": "center",       # "center" | "random"

    # --- Spatial ---
    "spatial_method": "uniform",       # "uniform" | "clustered"
    "n_clusters": 4,
    "cluster_std": 8.0,

    # --- Demande ---
    "demand_method": "uniform",
    "demand_range": (1, 9),

    # --- Capacité ---
    # CONTRAINTE DE FAISABILITÉ : capacity >= demand_range[1], sinon un client
    # dont la demande dépasse la capacité serait impossible à servir et
    # l'épisode RL ne pourrait jamais se terminer. Vérifié par validate_instance.
    "capacity": 30,

    # --- Dynamisme (le cœur du "Temps Réel" du sujet) ---
    "dynamic": True,
    "degree_of_dynamism": 0.5,        # DoD : proportion de clients dynamiques
    "arrival_distribution": "poisson", # "uniform" | "poisson"
    "planning_horizon": (0, 480),      # minutes (8h de service)

    # --- Temps de service ---
    "service_time_range": (5, 15),     # minutes passées chez le client

    # --- Vitesse (conversion distance -> temps) ---
    "velocity": 1.0,                    # unité de distance / minute
}


# ---------------------------------------------------------------------------
# Environnement RL
# ---------------------------------------------------------------------------

ENV_CONFIG = {
    # Pénalité appliquée à un client encore non servi si l'épisode est coupé
    # par le garde-fou max_steps. Ne devrait jamais se déclencher en pratique
    # (le masquage garantit qu'un client reste toujours servable) — c'est un
    # filet de sécurité, pas un levier de réglage.
    "unserved_penalty": 200.0,
    "max_steps_factor": 4,              # max_steps = n_nodes * ce facteur

    # --- Congestion (cf. data/travel_time.py) ---
    # "off"            : temps = distance / vitesse (comportement historique)
    # "faithful"       : modèle de Yahya à l'identique (effet mesuré ~0.2%)
    # "normalized"     : idem avec gaussiennes normalisées (effet ~1%)
    # "multiplicative" : ralentissement proportionnel (effet ~75% en pointe)
    # Défaut "off" pour ne pas invalider les résultats déjà obtenus.
    "congestion_mode": "off",
    "congestion_params": None,          # None -> CONGESTION_PARAMS par défaut

    # --- Objectif d'optimisation (cf. rl/env.py::final_reward) ---
    # "distance" : minimise la distance parcourue  [défaut historique]
    # "time"     : minimise la durée totale (makespan) -> seul mode où la
    #              congestion influence réellement les décisions apprises
    # "combined" : distance + time_weight * makespan
    "objective": "distance",
    "time_weight": 1.0,
}


# ---------------------------------------------------------------------------
# Modèle GNN + politique
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "embed_dim": 128,
    "n_heads": 8,
    "n_gnn_layers": 3,
    "tanh_clip": 10.0,                  # clipping des logits (stabilité)
}


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 100,
    "batches_per_epoch": 20,
    "lr": 1e-4,
    "grad_clip": 1.0,
    "entropy_coef": 0.01,

    # --- Randomisation de domaine (cf. data/domain_randomization.py) ---
    "domain_randomization": True,

    # --- Baseline figée (cf. rl/train.py pour la justification) ---
    "baseline_update_period": 1,        # tenter la mise à jour toutes les N époques
    "baseline_eval_size": 256,          # taille du jeu d'évaluation de la baseline
    "baseline_significance": 0.05,      # seuil du test t apparié

    "seed": 0,
    "device": "cpu",                    # "cuda" sur Colab avec GPU
}
