"""
Validation d'une instance DVRP.

Principe : la validation DÉTECTE et RAPPORTE, elle ne corrige jamais
silencieusement. Elle retourne (stats, violations) ; c'est à l'appelant de
décider quoi en faire.

Contraintes vérifiées :
    demand_i <= capacity                 (sinon client jamais servable)
    demand_0 == 0                        (dépôt sans demande)
    0 <= r_i <= horizon                   (apparition dans l'horizon)
    r_i == 0 pour les clients statiques
    service_i >= 0
    dist_matrix symétrique, diagonale nulle
"""

import numpy as np


def validate_instance(instance):
    violations = []
    stats = {}

    n = instance.n_total()
    demands = np.asarray(instance.demands)
    appear = np.asarray(instance.appearance_times)
    is_dyn = np.asarray(instance.is_dynamic)
    service = np.asarray(instance.service_times)
    dist = np.asarray(instance.dist_matrix)

    horizon = instance.metadata.get("planning_horizon", (0, 480))[1]

    stats["n_customers"] = instance.n_customers()
    stats["capacity"] = instance.capacity
    stats["total_demand"] = instance.total_demand()
    stats["min_vehicles"] = instance.min_vehicles()
    stats["n_dynamic"] = instance.n_dynamic()

    # --- Faisabilité capacité (critique pour la terminaison des épisodes RL) ---
    too_big = np.where(demands > instance.capacity)[0]
    for i in too_big:
        violations.append({
            "type": "demand_exceeds_capacity", "node": int(i),
            "detail": f"demande={demands[i]} > capacité={instance.capacity}",
        })

    if demands[0] != 0:
        violations.append({
            "type": "depot_has_demand", "node": 0,
            "detail": f"demande du dépôt = {demands[0]}, attendu 0",
        })

    # --- Cohérence temporelle des arrivées ---
    bad_appear = np.where((appear < 0) | (appear > horizon))[0]
    for i in bad_appear:
        violations.append({
            "type": "appearance_outside_horizon", "node": int(i),
            "detail": f"r={appear[i]}, horizon={horizon}",
        })

    static_with_time = np.where((~is_dyn) & (appear > 0))[0]
    for i in static_with_time:
        violations.append({
            "type": "static_customer_has_appearance", "node": int(i),
            "detail": f"client statique avec r={appear[i]}",
        })

    # --- Temps de service ---
    neg_service = np.where(service < 0)[0]
    for i in neg_service:
        violations.append({
            "type": "negative_service_time", "node": int(i),
            "detail": f"service={service[i]}",
        })

    # --- Matrice de distances ---
    if not np.allclose(dist, dist.T, atol=1e-6):
        violations.append({"type": "dist_matrix_not_symmetric", "detail": "d[i,j] != d[j,i]"})
    if not np.allclose(np.diag(dist), 0, atol=1e-6):
        violations.append({"type": "dist_diagonal_not_zero", "detail": "d[i,i] != 0"})

    # --- Statistiques descriptives ---
    if instance.n_customers() > 0:
        cust_demands = demands[1:]
        stats["demand_mean"] = float(cust_demands.mean())
        stats["demand_std"] = float(cust_demands.std())
        stats["dist_to_depot_mean"] = float(dist[0, 1:].mean())
        dyn_appear = appear[1:][is_dyn[1:]]
        if len(dyn_appear) > 0:
            stats["appearance_mean"] = float(dyn_appear.mean())
    stats.update(instance.metadata.get("dod_stats", {}))

    stats["n_violations"] = len(violations)
    return stats, violations


def validate_dataset(instances, verbose=True):
    """Valide un dataset entier, retourne le nombre total de violations."""
    total = 0
    by_type = {}
    for idx, inst in enumerate(instances):
        _, viols = validate_instance(inst)
        total += len(viols)
        for v in viols:
            by_type[v["type"]] = by_type.get(v["type"], 0) + 1
            if verbose and total <= 10:
                print(f"  instance {idx} : [{v['type']}] {v['detail']}")

    if verbose:
        print(f"\n{len(instances)} instances validées — {total} violation(s)")
        if by_type:
            for t, c in sorted(by_type.items()):
                print(f"  {t}: {c}")
        else:
            print("  Aucune violation détectée.")
    return total, by_type


def print_validation_report(stats, violations, max_show=10):
    print("=" * 52)
    print("RAPPORT DE VALIDATION")
    print("=" * 52)
    for k, v in stats.items():
        print(f"{k:26s}: {v}")
    if violations:
        print(f"\n{len(violations)} violation(s) :")
        for v in violations[:max_show]:
            node = f" (nœud {v['node']})" if "node" in v else ""
            print(f"  - [{v['type']}]{node} {v['detail']}")
    else:
        print("\nAucune violation détectée.")
