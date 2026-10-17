"""
Randomisation de domaine pour l'entraînement RL.

Sans ceci, chaque batch d'entraînement utilise la MÊME configuration
(toujours 20 clients, toujours uniforme, toujours DoD=0.5...). Le modèle
optimise alors une politique très bien réglée pour cette configuration
précise, mais rien ne garantit qu'elle reste bonne ailleurs.

Avec la randomisation de domaine, chaque batch pioche une configuration
différente dans des plages définies (taille, distribution spatiale, degré
de dynamisme, demande, capacité). Le modèle est ainsi poussé à apprendre
des règles de décision qui marchent EN MOYENNE sur toute une famille de
problèmes, pas une seule instance de cette famille.

IMPORTANT — ce que ça ne résout PAS : ceci reste 100% synthétique. Ça
réduit la dépendance à une configuration synthétique particulière, ça ne
comble pas l'écart avec des données réelles (topologie routière, demande
réelle, etc. — cf. CAHIER_DES_CHARGES.md, section limites).
"""

import numpy as np

DOMAIN_RANGES = {
    "n_customers": (10, 30),                    # taille de l'instance
    "spatial_method": ["uniform", "clustered"],
    "degree_of_dynamism": (0.2, 0.7),
    "arrival_distribution": ["uniform", "poisson"],
    "demand_range_max": (6, 12),                 # borne basse fixée à 1
    "n_clusters": (2, 6),                         # utilisé seulement si "clustered"
    "capacity_factor": (3, 5),                    # capacité = demand_max * facteur
}


def sample_domain_config(rng, ranges=None):
    """
    Tire une configuration d'instance aléatoire dans les plages données.
    Retourne un dict directement utilisable en **kwargs de generate_instance.

    rng : np.random.default_rng dédié à l'échantillonnage de domaine
          (séparé du rng de génération d'instance elle-même, pour ne pas
          mélanger les deux sources de hasard).
    """
    r = ranges or DOMAIN_RANGES

    n_customers = int(rng.integers(r["n_customers"][0], r["n_customers"][1] + 1))
    spatial_method = str(rng.choice(r["spatial_method"]))
    dod = float(rng.uniform(*r["degree_of_dynamism"]))
    arrival_dist = str(rng.choice(r["arrival_distribution"]))
    demand_max = int(rng.integers(r["demand_range_max"][0], r["demand_range_max"][1] + 1))
    capacity_factor = int(rng.integers(*r["capacity_factor"]))

    # Capacité cohérente avec la demande max : garde-fou de faisabilité de
    # generate_instance (demand_range[1] <= capacity) toujours respecté par
    # construction, tout en gardant un nombre de tournées raisonnable.
    capacity = demand_max * capacity_factor

    config = {
        "n_customers": n_customers,
        "spatial_method": spatial_method,
        "degree_of_dynamism": dod,
        "arrival_distribution": arrival_dist,
        "demand_range": (1, demand_max),
        "capacity": capacity,
    }
    if spatial_method == "clustered":
        config["n_clusters"] = int(rng.integers(r["n_clusters"][0], r["n_clusters"][1] + 1))

    return config
