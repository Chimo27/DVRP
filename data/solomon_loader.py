"""
Chargement d'une instance Solomon (.txt, format standard) et conversion
vers notre structure Instance.

POINT IMPORTANT : les instances Solomon sont conçues pour le VRPTW (avec
fenêtres de temps). Notre projet exclut volontairement les fenêtres de
temps (cf. CAHIER_DES_CHARGES.md). On charge donc uniquement les positions,
demandes et la capacité — les fenêtres de temps du fichier sont IGNORÉES.

Conséquence à ne jamais perdre de vue : l'optimum "officiel" publié pour une
instance Solomon (ex. C101, R101) est l'optimum du problème AVEC fenêtres.
Retirer une contrainte ne peut que faire baisser (ou laisser égal)
l'optimum réel — comparer notre solution à ce chiffre publié serait donc
une comparaison invalide, à l'avantage artificiel de notre modèle. La bonne
pratique : recalculer l'optimum de la version SANS fenêtres avec un
solveur exact (cf. exact_solver.py), sur le même sous-problème.

Format attendu (standard Solomon, cf. web.cba.neu.edu/~msolomon) :

    <nom>

    VEHICLE
    NUMBER     CAPACITY
      K          Q

    CUSTOMER
    CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
        0        x0       y0      0         0         999        0
        1        x1       y1      d1       ...
        ...
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.instance import Instance


def load_solomon_instance(filepath, n_customers=None, capacity_override=None):
    """
    Lit un fichier Solomon et retourne une Instance CVRP statique (sans
    fenêtres de temps, sans dynamisme).

    n_customers : si fourni, ne garde que les n_customers premiers clients
        du fichier (utile pour rester dans une taille résoluble EXACTEMENT
        par le solveur de référence — cf. exact_solver.py, au-delà d'une
        quinzaine de clients un solveur exact devient trop lent).
    capacity_override : ignore la capacité du fichier si fourni.
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines()]

    name = lines[0]

    veh_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("VEHICLE"))
    capacity = int(lines[veh_idx + 2].split()[1])
    if capacity_override is not None:
        capacity = capacity_override

    cust_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("CUSTOMER"))
    data_start = cust_idx + 2  # saute l'en-tête de colonnes

    rows = []
    for l in lines[data_start:]:
        if not l.strip():
            continue
        parts = l.split()
        if len(parts) < 7:
            continue
        rows.append([float(x) for x in parts[:7]])
    rows = np.array(rows)  # colonnes : id, x, y, demande, ready, due, service

    if n_customers is not None:
        rows = rows[: n_customers + 1]  # +1 pour le dépôt (ligne 0)

    locations = rows[:, 1:3]
    demands = rows[:, 3]

    n = len(rows) - 1
    dist_matrix = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :], axis=-1
    )

    return Instance(
        locations=locations,
        demands=demands,
        capacity=capacity,
        appearance_times=np.zeros(len(rows)),   # statique : personne n'apparaît en cours de route
        is_dynamic=np.zeros(len(rows), dtype=bool),
        service_times=rows[:, 6],
        dist_matrix=dist_matrix,
        metadata={
            "source": "solomon", "name": name, "n_customers": n,
            "note": "fenêtres de temps du fichier original IGNORÉES (hors périmètre du projet)",
        },
    )
