"""
Inférence sur une seule instance — pas d'évaluation, pas de moyennes.

    from rl.infer import solve_instance, print_solution

    tours, distance = solve_instance(policy, instance)
    print_solution(tours, instance)
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.env import DVRPEnv


@torch.no_grad()
def solve_instance(policy, instance, device="cpu", greedy=True):
    """
    Fait tourner la politique (déjà entraînée) sur UNE instance, en pur
    mode inférence (aucun gradient, aucun optimiseur).

    Retourne :
        tours    : liste de tournées, chaque tournée = liste d'indices
                   clients (1..n), un passage par le dépôt = une tournée
        distance : distance totale parcourue (toutes tournées confondues)
    """
    policy.eval()
    # Même chemin de code que l'entraînement et l'évaluation (rollout) :
    # même observation, même encodage événementiel, même masque.
    env = DVRPEnv([instance], device=device, information=policy.information)
    with torch.no_grad():
        policy.rollout(env, greedy=greedy)
    route = [0] + [int(a[0]) for a in policy.last_actions if int(a[0]) >= 0]

    tours = _route_to_tours(route)
    distance = env.total_distance.item()
    return tours, distance


def _route_to_tours(route):
    """
    Convertit la séquence brute de nœuds visités (dépôt=0) en liste de
    tournées séparées. Fusionne les répétitions consécutives (une "attente
    sur place" au dépôt produit deux 0 d'affilée sans déplacement réel) et
    ignore les passages au dépôt qui ne débouchent sur aucune tournée.
    """
    dedup = [route[0]]
    for n in route[1:]:
        if n != dedup[-1]:
            dedup.append(n)

    tours, current = [], []
    for n in dedup:
        if n == 0:
            if current:
                tours.append(current)
                current = []
        else:
            current.append(n)
    if current:
        tours.append(current)
    return tours


def _tour_distance(instance, tour):
    nodes = [0] + tour + [0]
    d = instance.dist_matrix
    return sum(d[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1))


def print_solution(tours, instance=None):
    """
    Affiche les tournées au format :
        Véhicule 1 : Dépôt → Client 7 → Client 12 → Client 3 → Dépôt
    """
    total = 0.0
    for i, tour in enumerate(tours, 1):
        path = " → ".join(["Dépôt"] + [f"Client {c}" for c in tour] + ["Dépôt"])
        if instance is not None:
            d = _tour_distance(instance, tour)
            total += d
            print(f"Véhicule {i} : {path}   (distance {d:.1f})")
        else:
            print(f"Véhicule {i} : {path}")

    if instance is not None:
        print(f"\n{len(tours)} véhicule(s) — distance totale : {total:.1f}")
