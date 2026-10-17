"""
Modèle de temps de trajet dépendant de l'heure (congestion).

Adapté du générateur de Yahya (SVRPBench / travel_time_generator.py), avec
trois différences d'implémentation nécessaires à notre projet :

  1. VECTORISÉ (torch) : notre environnement traite un lot d'instances en
     parallèle ; l'original est scalaire.
  2. REPRODUCTIBLE : l'original appelle random.* et np.random.* globaux.
     Ici tout le hasard passe par un torch.Generator explicite.
  3. DEUX MODES DE CALIBRATION (voir ci-dessous).

--------------------------------------------------------------------------
POURQUOI DEUX MODES — mesure effectuée sur le code original
--------------------------------------------------------------------------
Dans l'original, `normal_distribution` est une DENSITÉ de probabilité : son
maximum vaut 1/(σ√2π) ≈ 0.0044 pour σ=90. Le terme de pic (×2) pèse donc
0.009 face à une base de 0.5. Mesures :

    time_factor sur 24h : min 0.5000, max 0.5089  ->  1.77 % de variation
    retard sur un trajet de 50 unités : 0.08 min  ->  0.16 % du temps
    accidents : ~1 tous les 6000 trajets

Autrement dit, le modèle original ne produit quasiment AUCUNE congestion
observable : les temps « avec congestion » sont indiscernables des temps
déterministes.

  mode="faithful"   reproduit exactement l'original (effet ~0, utile pour
                     comparer / justifier le constat)
  mode="normalized" remplace la densité par une gaussienne normalisée à 1
                     au sommet, ce qui donne un facteur allant de 0.5 hors
                     pointe à ~2.5 en pointe (variation ×5), donc une
                     congestion réellement visible.

--------------------------------------------------------------------------
UNE LIMITE STRUCTURELLE DE LA FORMULE ADDITIVE (mesurée)
--------------------------------------------------------------------------
Dans l'original, le retard vaut 0.25 × time_factor × (1 − exp(−d/50)). Le
terme de distance SATURE à 1, donc le retard est plafonné :

    retard_max = 0.25 × 2.5 × 1.0 = 0.625 min, quelle que soit la distance

Mesure (mode normalized, heure de pointe) :

    distance   10 -> retard 0.14 min (1.41 % du trajet)
    distance   50 -> retard 0.49 min (0.99 %)
    distance  500 -> retard 0.78 min (0.16 %)
    distance 1000 -> retard 0.78 min (0.08 %)

Un trajet long subit donc proportionnellement MOINS de congestion qu'un
trajet court — l'inverse du comportement physique attendu (traverser une
heure de pointe pendant longtemps devrait coûter davantage).

  mode="multiplicative" applique le ralentissement au TEMPS DE TRAJET
                         lui-même : temps = (d / v) × m(heure), avec m
                         allant de 1.0 hors pointe à `peak_multiplier` en
                         pointe. C'est la formulation usuelle du VRP
                         dépendant du temps (TDVRP), et la seule des trois
                         où la congestion a un effet proportionnel.

Aucun des trois n'est « le bon » a priori : c'est un choix de modélisation
à valider (données réelles / littérature).

--------------------------------------------------------------------------
HORIZON ET HEURE DE LA JOURNÉE
--------------------------------------------------------------------------
Les pics de l'original sont exprimés en minutes depuis minuit (480 = 8h,
1020 = 17h). Notre horizon de planification est une journée de service de
480 minutes comptée à partir de 0. On mappe donc :

    heure_horloge = day_start_minutes + temps_environnement

avec day_start_minutes = 480 par défaut (service démarrant à 8h). Une
journée de 480 min couvre alors 8h -> 16h : tout le pic du matin, et
l'approche du pic du soir.
"""

import math
import torch

# --- Paramètres du modèle (repris de l'original, tous surchargeables) ---
CONGESTION_PARAMS = {
    "morning_peak": 480.0,      # 8h00, en minutes depuis minuit
    "morning_std": 90.0,
    "evening_peak": 1020.0,     # 17h00
    "evening_std": 90.0,
    "base_factor": 0.5,         # facteur hors pointe
    "peak_weight": 2.0,         # poids des pics
    "delay_coef": 0.25,         # coefficient de retard de base
    "distance_scale": 50.0,     # échelle de saturation du retard avec la distance
    "noise_mu": 0.0,            # bruit log-normal : moyenne de base
    "noise_mu_rush": 0.1,       #                    surcroît en pointe
    "noise_sigma": 0.3,         #                    écart-type de base
    "noise_sigma_rush": 0.2,    #                    surcroît en pointe
    "accident_peak": 1260.0,    # 21h00
    "accident_std": 120.0,
    "accident_rate": 0.05,
    "accident_min_delay": 30.0,
    "accident_max_delay": 120.0,
    "day_start_minutes": 480.0,  # décalage horizon -> heure d'horloge
    "peak_multiplier": 1.8,      # mode "multiplicative" : ralentissement max en pointe
}


def _gaussian(x, mean, std, normalized):
    """
    normalized=False : densité de probabilité (comportement original,
        sommet = 1/(std*sqrt(2*pi)), donc très petit)
    normalized=True  : gaussienne de sommet 1
    """
    g = torch.exp(-((x - mean) ** 2) / (2 * std ** 2))
    if not normalized:
        g = g / (std * math.sqrt(2 * math.pi))
    return g


def time_factor(clock_time, params, normalized):
    """Facteur de ralentissement lié à l'heure (pics matin / soir)."""
    p = params
    morning = _gaussian(clock_time, p["morning_peak"], p["morning_std"], normalized)
    evening = _gaussian(clock_time, p["evening_peak"], p["evening_std"], normalized)
    return p["base_factor"] + p["peak_weight"] * (morning + evening)


def _accident_delay(distance, clock_time, p, normalized, dev, generator):
    """Retard lié aux accidents : rares (loi de Poisson) mais coûteux."""
    rate = p["accident_rate"] * _gaussian(
        clock_time, p["accident_peak"], p["accident_std"], normalized
    )
    rate = torch.clamp(rate, min=0.0)
    n_acc = torch.poisson(rate.expand_as(distance).contiguous(), generator=generator)
    u = torch.rand(distance.shape, device=dev, generator=generator)
    spread = p["accident_max_delay"] - p["accident_min_delay"]
    return n_acc * (p["accident_min_delay"] + u * spread)


def compute_travel_time(
    distance, clock_time, params=None, mode="normalized",
    velocity=1.0, generator=None, enable_accidents=True,
):
    """
    Temps de trajet = distance / vitesse + retard(distance, heure).

    distance, clock_time : tenseurs de même forme (batch)
    mode      : "faithful" | "normalized" | "off"
    generator : torch.Generator pour la reproductibilité du bruit

    Retourne un tenseur de même forme que `distance`.
    """
    p = {**CONGESTION_PARAMS, **(params or {})}
    base_time = distance / velocity

    if mode == "off":
        return base_time

    dev = distance.device

    # --- Mode multiplicatif : le ralentissement s'applique au temps lui-même ---
    if mode == "multiplicative":
        rush = (_gaussian(clock_time, p["morning_peak"], p["morning_std"], True)
                + _gaussian(clock_time, p["evening_peak"], p["evening_std"], True))
        rush = torch.clamp(rush, max=1.0)          # deux pics peuvent se chevaucher
        mult = 1.0 + (p["peak_multiplier"] - 1.0) * rush

        sigma = p["noise_sigma"] + p["noise_sigma_rush"] * rush
        noise = torch.exp(sigma * torch.randn(distance.shape, device=dev, generator=generator)
                           - 0.5 * sigma ** 2)      # médiane 1, pas de biais moyen
        travel = base_time * mult * noise

        if enable_accidents:
            travel = travel + _accident_delay(distance, clock_time, p, True, dev, generator)
        return travel

    normalized = (mode == "normalized")

    # --- Retard de base : croît avec la distance, saturant ---
    tf = time_factor(clock_time, p, normalized)
    distance_factor = 1.0 - torch.exp(-distance / p["distance_scale"])
    base_delay = p["delay_coef"] * tf * distance_factor

    # --- Bruit log-normal, plus dispersé en heure de pointe ---
    rush = (_gaussian(clock_time, p["morning_peak"], p["morning_std"], normalized)
            + _gaussian(clock_time, p["evening_peak"], p["evening_std"], normalized))
    mu = p["noise_mu"] + p["noise_mu_rush"] * rush
    sigma = p["noise_sigma"] + p["noise_sigma_rush"] * rush
    normal_noise = torch.randn(distance.shape, device=dev, generator=generator)
    noise = torch.exp(mu + sigma * normal_noise)

    delay = base_delay * noise

    # --- Accidents : rares, mais coûteux quand ils surviennent ---
    if enable_accidents:
        delay = delay + _accident_delay(distance, clock_time, p, normalized, dev, generator)

    return base_time + delay


def env_time_to_clock(env_time, params=None):
    """Convertit le temps de l'environnement en heure d'horloge (minutes depuis minuit)."""
    p = {**CONGESTION_PARAMS, **(params or {})}
    return env_time + p["day_start_minutes"]
