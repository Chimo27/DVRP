"""
Orchestrateur : assemble les briques de components.py en une Instance
complète, entièrement reproductible à partir d'une seed.

Pipeline :
    seed -> rng unique
        1. positions (spatial)
        2. demandes
        3. arrivées dynamiques (r_i)
        4. temps de service
        5. matrice de distances
    -> Instance
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import INSTANCE_CONFIG
from data.instance import Instance
from data.components import (
    generate_positions, generate_demands,
    generate_arrivals, generate_service_times,
)


def generate_instance(seed=None, **overrides):
    """
    Génère une instance DVRP. Tout paramètre de INSTANCE_CONFIG peut être
    surchargé par mot-clé, ex :

        generate_instance(seed=42, n_customers=50, spatial_method="clustered")
    """
    cfg = {**INSTANCE_CONFIG, **overrides}
    rng = np.random.default_rng(seed)

    n = cfg["n_customers"]
    capacity = cfg["capacity"]

    # --- Garde-fou de faisabilité (cf. config.py) ---
    if cfg["demand_range"][1] > capacity:
        raise ValueError(
            f"demande max ({cfg['demand_range'][1]}) > capacité ({capacity}) : "
            "un tel client serait impossible à servir, l'épisode RL ne pourrait "
            "jamais se terminer."
        )

    # --- 1. Spatial ---
    depot, customers = generate_positions(
        n, rng, method=cfg["spatial_method"],
        coord_range=cfg["coord_range"], depot_position=cfg["depot_position"],
        n_clusters=cfg["n_clusters"], cluster_std=cfg["cluster_std"],
    )
    locations = np.vstack([depot, customers])

    # --- 2. Demandes ---
    demands = generate_demands(
        n, rng, method=cfg["demand_method"], demand_range=cfg["demand_range"],
    )
    demands = np.concatenate([[0], demands]).astype(float)

    # --- 3. Arrivées dynamiques ---
    if cfg["dynamic"]:
        is_dyn, appearance, dod_stats = generate_arrivals(
            n, rng,
            degree_of_dynamism=cfg["degree_of_dynamism"],
            distribution=cfg["arrival_distribution"],
            planning_horizon=cfg["planning_horizon"],
        )
    else:
        is_dyn = np.zeros(n, dtype=bool)
        appearance = np.zeros(n)
        dod_stats = {"distribution": "none", "dod_target": 0.0,
                     "dod_actual": 0.0, "effective_dod": 0.0}

    is_dynamic = np.concatenate([[False], is_dyn])
    appearance_times = np.concatenate([[0.0], appearance])

    # --- 4. Temps de service ---
    service = generate_service_times(
        n, rng, service_time_range=cfg["service_time_range"],
    )
    service_times = np.concatenate([[0.0], service])

    # --- 5. Distances (euclidiennes) ---
    dist_matrix = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :], axis=-1
    )

    return Instance(
        locations=locations,
        demands=demands,
        capacity=capacity,
        appearance_times=appearance_times,
        is_dynamic=is_dynamic,
        service_times=service_times,
        dist_matrix=dist_matrix,
        metadata={
            "seed": seed,
            "n_customers": n,
            "spatial_method": cfg["spatial_method"],
            "demand_method": cfg["demand_method"],
            "arrival_distribution": cfg["arrival_distribution"],
            "planning_horizon": cfg["planning_horizon"],
            "velocity": cfg["velocity"],
            "dod_stats": dod_stats,
            "config": {k: v for k, v in cfg.items()},
        },
    )


def generate_dataset(n_instances, base_seed=0, **overrides):
    """Génère une liste d'instances avec des seeds consécutives (reproductible)."""
    return [generate_instance(seed=base_seed + i, **overrides)
            for i in range(n_instances)]


def save_dataset(instances, filepath):
    """Sauvegarde un dataset au format .npz compressé."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    payload = {
        "locations": np.array([i.locations for i in instances]),
        "demands": np.array([i.demands for i in instances]),
        "capacity": np.array([i.capacity for i in instances]),
        "appearance_times": np.array([i.appearance_times for i in instances]),
        "is_dynamic": np.array([i.is_dynamic for i in instances]),
        "service_times": np.array([i.service_times for i in instances]),
        "dist_matrix": np.array([i.dist_matrix for i in instances]),
        "metadata": np.array([i.metadata for i in instances], dtype=object),
    }
    np.savez_compressed(filepath, **payload)
    return filepath


def load_dataset(filepath):
    """Recharge un dataset .npz en liste d'Instance."""
    d = np.load(filepath, allow_pickle=True)
    n = len(d["locations"])
    return [
        Instance(
            locations=d["locations"][i],
            demands=d["demands"][i],
            capacity=int(d["capacity"][i]),
            appearance_times=d["appearance_times"][i],
            is_dynamic=d["is_dynamic"][i],
            service_times=d["service_times"][i],
            dist_matrix=d["dist_matrix"][i],
            metadata=d["metadata"][i],
        )
        for i in range(n)
    ]
