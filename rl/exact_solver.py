"""
Solveur de référence pour une instance CVRP — sert de point de comparaison
"vérité terrain" pour le modèle RL+GNN.

[RÉVISION] solve_exact utilise désormais CP-SAT, qui PROUVE l'optimalité
(statut OPTIMAL). L'ancienne fonction, renommée solve_gls, utilise la
recherche locale guidée d'OR-Tools, qui ne fournit aucune preuve.

NOTE D'ORIGINE : OR-Tools ne garantit l'OPTIMUM PROUVÉ que sur de
petites instances (typiquement <= 12-15 clients dans un temps raisonnable).
Au-delà, il retourne la meilleure solution trouvée dans le temps imparti,
SANS garantie d'optimalité. La fonction retourne explicitement un champ
`proven_optimal` — ne jamais afficher "optimal" sans avoir vérifié ce champ.
"""

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def solve_gls(instance, time_limit_s=30, scale=1000):
    """
    Résout l'instance CVRP avec OR-Tools (recherche exacte pour les petites
    instances, best-effort avec preuve d'optimalité si le temps le permet).

    scale : les distances sont converties en entiers (OR-Tools travaille en
        entiers) ; scale=1000 garde 3 décimales de précision.

    Retourne : (distance_totale, tours, proven_optimal)
        tours : liste de tournées, même format que rl/infer.solve_instance
        proven_optimal : True seulement si OR-Tools a prouvé l'optimalité
            (pas juste trouvé une solution avant la limite de temps)
    """
    n = instance.n_total()
    dist_int = (instance.dist_matrix * scale).astype(int)
    demands = instance.demands.astype(int)
    capacity = int(instance.capacity)

    # Nombre de véhicules : borne large (le solveur n'est pas obligé de
    # tous les utiliser), cohérente avec la demande totale.
    n_vehicles = max(1, int(np.ceil(demands.sum() / capacity)) + 2)

    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)  # dépôt = index 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        i, j = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        return int(dist_int[i, j])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_idx):
        return int(demands[manager.IndexToNode(from_idx)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [capacity] * n_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools n'a trouvé aucune solution dans le temps imparti")

    # OR-Tools ne renvoie pas explicitement "j'ai prouvé l'optimum" pour la
    # métaheuristique guidée. Preuve d'optimalité fiable seulement pour la
    # stratégie exacte (petites instances, PATH_CHEAPEST_ARC seul suffit
    # souvent à trouver ET prouver l'optimum en dessous de ~12 clients).
    # [révision] La recherche locale guidée ne prouve jamais l'optimalité :
    # l'ancien indicateur (n <= 12 et statut SUCCESS) signifiait seulement
    # qu'une solution avait été trouvée. Utiliser solve_exact (CP-SAT) pour
    # une preuve.
    proven_optimal = False

    tours = []
    total_distance = 0.0
    for v in range(n_vehicles):
        idx = routing.Start(v)
        tour = []
        route_dist = 0
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                tour.append(node)
            next_idx = solution.Value(routing.NextVar(idx))
            route_dist += routing.GetArcCostForVehicle(idx, next_idx, v)
            idx = next_idx
        if tour:
            tours.append(tour)
            total_distance += route_dist

    return total_distance / scale, tours, proven_optimal


def solve_exact(instance, time_limit_s=120, scale=1000):
    """CVRP statique résolu à l'optimum prouvé par CP-SAT (révision).

    Modèle : circuits multiples passant par le dépôt (AddMultipleCircuit),
    charges cumulées de type MTZ. Retourne (distance, None, proven_optimal),
    proven_optimal étant vrai seulement si CP-SAT renvoie le statut OPTIMAL.
    """
    from ortools.sat.python import cp_model
    n = instance.n_total()
    D = (instance.dist_matrix * scale).astype(int)
    q = instance.demands.astype(int)
    Q = int(instance.capacity)
    m = cp_model.CpModel()
    x = {(i, j): m.NewBoolVar(f"x{i}_{j}") for i in range(n) for j in range(n) if i != j}
    m.AddMultipleCircuit([(i, j, v) for (i, j), v in x.items()])
    u = [m.NewIntVar(0, Q, f"u{i}") for i in range(n)]
    for i in range(1, n):
        m.Add(u[i] >= int(q[i]))
        m.Add(u[i] == int(q[i])).OnlyEnforceIf(x[0, i])
        for j in range(1, n):
            if i != j:
                m.Add(u[j] == u[i] + int(q[j])).OnlyEnforceIf(x[i, j])
    m.Minimize(sum(int(D[i, j]) * v for (i, j), v in x.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT n'a trouvé aucune solution")
    return solver.ObjectiveValue() / scale, None, status == cp_model.OPTIMAL
