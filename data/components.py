"""
Briques élémentaires de génération d'instances.

Chaque brique expose une interface commune et n'utilise QUE le `rng` fourni
(np.random.default_rng), jamais np.random.* directement : toute la
génération est reproductible à partir d'une seed unique.

    generate_positions(n_customers, rng, method, **params) -> (depot, customers)
    generate_demands(n_customers, rng, method, **params)   -> demands
    generate_arrivals(n_customers, rng, **params)          -> (is_dynamic, r_i, stats)
    generate_service_times(n_customers, rng, **params)     -> service_times
"""

import numpy as np


# ===========================================================================
# 1. SPATIAL — où sont les clients
# ===========================================================================

def _place_depot(rng, coord_range, depot_position):
    if depot_position == "center":
        return np.array([(coord_range[0] + coord_range[1]) / 2] * 2)
    if depot_position == "random":
        return rng.uniform(*coord_range, size=2)
    return np.array(depot_position, dtype=float)


def generate_positions(
    n_customers, rng, method="uniform",
    coord_range=(0, 100), depot_position="center",
    n_clusters=4, cluster_std=8.0, **kwargs,
):
    """
    method="uniform"   : clients tirés uniformément sur toute la carte.
        Hypothèse neutre, convention des benchmarks CVRP classiques.

    method="clustered" : clients regroupés en gaussiennes autour de
        n_clusters centres tirés uniformément. Approxime des zones
        résidentielles / quartiers, plus proche d'une carte réelle.
    """
    depot = _place_depot(rng, coord_range, depot_position)

    if method == "uniform":
        customers = rng.uniform(*coord_range, size=(n_customers, 2))

    elif method == "clustered":
        centers = rng.uniform(*coord_range, size=(n_clusters, 2))
        assignments = rng.integers(0, n_clusters, size=n_customers)
        customers = np.zeros((n_customers, 2))
        for i, c in enumerate(assignments):
            pos = rng.normal(loc=centers[c], scale=cluster_std, size=2)
            customers[i] = np.clip(pos, coord_range[0], coord_range[1])

    else:
        raise ValueError("method doit être 'uniform' ou 'clustered'")

    return depot, customers


# ===========================================================================
# 2. DEMANDE — combien commande chaque client
# ===========================================================================

def generate_demands(n_customers, rng, method="uniform", demand_range=(1, 9), **kwargs):
    """
    method="uniform" : loi uniforme discrète sur [demand_range].
        Convention dominante des benchmarks CVRP et de la littérature RL-VRP.
    """
    if method != "uniform":
        raise ValueError("seule la méthode 'uniform' est disponible actuellement")
    return rng.integers(demand_range[0], demand_range[1] + 1, size=n_customers)


# ===========================================================================
# 3. ARRIVÉES DYNAMIQUES — qui apparaît en cours de journée
# ===========================================================================

def generate_arrivals(
    n_customers, rng,
    degree_of_dynamism=0.5, distribution="poisson",
    planning_horizon=(0, 480), reveal_deadline_ratio=0.75, **kwargs,
):
    """
    Détermine quels clients sont connus dès le départ (statiques) et lesquels
    apparaissent en cours de journée (dynamiques), avec leur instant r_i.

    degree_of_dynamism (DoD) : proportion de clients dynamiques
        (Lund, Madsen & Rygaard, 1996).

    distribution="uniform" : r_i tiré uniformément sur la fenêtre d'arrivée.
    distribution="poisson" : processus de Poisson homogène (inter-arrivées
        exponentielles) — modélise des commandes qui tombent "au fil de
        l'eau", plus réaliste qu'un tirage indépendant.

    reveal_deadline_ratio : les arrivées sont confinées à la première
        fraction de l'horizon (ex. 0.75 = 6h sur 8h). Sans cela, un client
        apparaissant à la toute dernière minute rendrait l'instance
        artificiellement pénible sans intérêt pour l'apprentissage.
    """
    is_dynamic = rng.uniform(0, 1, size=n_customers) < degree_of_dynamism
    appearance = np.zeros(n_customers)
    dyn_idx = np.where(is_dynamic)[0]
    n_dyn = len(dyn_idx)

    h_start, h_end = planning_horizon
    arrival_end = h_start + (h_end - h_start) * reveal_deadline_ratio

    if n_dyn > 0:
        if distribution == "uniform":
            times = rng.uniform(h_start, arrival_end, size=n_dyn)

        elif distribution == "poisson":
            span = arrival_end - h_start
            rate = n_dyn / span
            times, t = [], h_start
            while len(times) < n_dyn and t < arrival_end:
                t += rng.exponential(1.0 / rate)
                if t <= arrival_end:
                    times.append(t)
            # Le processus peut s'arrêter avant d'avoir généré n_dyn points
            # (queue de l'horizon) : on complète uniformément pour respecter
            # exactement le DoD demandé.
            while len(times) < n_dyn:
                times.append(rng.uniform(h_start, arrival_end))
            times = np.array(times[:n_dyn])

        else:
            raise ValueError("distribution doit être 'uniform' ou 'poisson'")

        appearance[dyn_idx] = times

    appearance = np.round(appearance, 1)

    span_total = h_end - h_start
    edd = float(np.mean(appearance) / span_total) if span_total > 0 else 0.0

    stats = {
        "distribution": distribution,
        "dod_target": degree_of_dynamism,
        "dod_actual": round(float(n_dyn / n_customers), 4) if n_customers else 0.0,
        # EDD : degré de dynamisme "effectif", pondéré par l'instant
        # d'apparition (un client révélé tard pèse plus sur la difficulté
        # réelle qu'un client révélé tôt).
        "effective_dod": round(edd, 4),
    }
    return is_dynamic, appearance, stats


# ===========================================================================
# 4. TEMPS DE SERVICE
# ===========================================================================

def generate_service_times(n_customers, rng, service_time_range=(5, 15), **kwargs):
    """Durée passée sur place chez chaque client (minutes)."""
    return rng.integers(service_time_range[0], service_time_range[1] + 1,
                         size=n_customers).astype(float)
