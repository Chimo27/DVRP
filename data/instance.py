"""
Représentation commune d'une instance DVRP (Dynamic Capacitated VRP).

Convention : l'index 0 correspond TOUJOURS au dépôt.
    locations[0]        = position du dépôt
    demands[0]          = 0
    appearance_times[0] = 0
    is_dynamic[0]       = False

Périmètre volontaire (cf. CAHIER_DES_CHARGES.md) : pas de fenêtres de temps.
Le champ `time_windows` est conservé à None pour permettre une extension
future sans casser le format, mais n'est utilisé par aucune brique actuelle.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class Instance:
    # --- Obligatoire ---
    locations: np.ndarray            # (n_total, 2)
    demands: np.ndarray              # (n_total,)
    capacity: int

    # --- Dynamisme ---
    appearance_times: np.ndarray     # (n_total,) : r_i, 0 si statique
    is_dynamic: np.ndarray           # (n_total,) bool

    # --- Temps ---
    service_times: np.ndarray        # (n_total,)
    dist_matrix: np.ndarray          # (n_total, n_total)

    # --- Extension future (non utilisé actuellement) ---
    time_windows: Optional[np.ndarray] = None

    # --- Traçabilité complète de la génération ---
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    def n_total(self) -> int:
        return len(self.locations)

    def n_customers(self) -> int:
        return len(self.locations) - 1

    def total_demand(self) -> int:
        return int(np.sum(self.demands))

    def min_vehicles(self) -> int:
        """Borne inférieure sur le nombre de tournées nécessaires."""
        return int(np.ceil(self.total_demand() / self.capacity))

    def n_dynamic(self) -> int:
        return int(np.sum(self.is_dynamic))

    def summary(self) -> str:
        lines = [
            f"Instance DVRP : {self.n_customers()} clients",
            f"  Capacité véhicule   : {self.capacity}",
            f"  Demande totale      : {self.total_demand()} "
            f"(min. {self.min_vehicles()} tournées)",
            f"  Clients dynamiques  : {self.n_dynamic()}/{self.n_customers()}",
        ]
        if "dod_stats" in self.metadata:
            s = self.metadata["dod_stats"]
            lines.append(f"  DoD réel / EDD      : {s['dod_actual']} / {s['effective_dod']}")
        if "spatial_method" in self.metadata:
            lines.append(f"  Méthode spatiale    : {self.metadata['spatial_method']}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation (pour np.savez_compressed)."""
        return {
            "locations": self.locations,
            "demands": self.demands,
            "capacity": self.capacity,
            "appearance_times": self.appearance_times,
            "is_dynamic": self.is_dynamic,
            "service_times": self.service_times,
            "dist_matrix": self.dist_matrix,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Instance":
        return Instance(
            locations=d["locations"],
            demands=d["demands"],
            capacity=int(d["capacity"]),
            appearance_times=d["appearance_times"],
            is_dynamic=d["is_dynamic"],
            service_times=d["service_times"],
            dist_matrix=d["dist_matrix"],
            metadata=d.get("metadata", {}),
        )
