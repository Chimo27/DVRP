# DVRP_RL_GNN

Optimisation de tournées logistiques **en temps réel** par Deep Reinforcement
Learning et Graph Neural Networks.

Voir [CAHIER_DES_CHARGES.md](CAHIER_DES_CHARGES.md) pour la spécification
complète (problème traité, choix techniques, protocole, limites).

---

## Installation

```bash
pip install torch torch_geometric numpy matplotlib scipy
```

## Utilisation en 4 étapes

```bash
python scripts/01_generate_dataset.py     # génère + valide un dataset
python scripts/02_visualize_dataset.py    # figures d'analyse du dataset
python scripts/03_train.py --epochs 100   # entraîne le modèle
python scripts/04_evaluate.py             # évalue vs heuristiques
```

## Utilisation en notebook (Colab)

**Activer le GPU** : `Runtime → Change runtime type → GPU`

```python
!pip install -q torch_geometric
%cd DVRP_RL_GNN
```

```python
# 1. Générer et inspecter une instance
from data.generator import generate_instance
from data.validation import validate_instance, print_validation_report
from viz.instance_viz import plot_instance_dashboard

inst = generate_instance(seed=42, n_customers=20)
print(inst.summary())
print_validation_report(*validate_instance(inst))
plot_instance_dashboard(inst)
```

```python
# 2. Entraîner
from rl.train import train

policy, history = train(
    n_customers=20,
    epochs=200,
    batch_size=128,
    device="cuda",
    checkpoint_path="outputs/checkpoints/policy.pt",
)
```

```python
# 3. Évaluer et visualiser
from rl.evaluate import evaluate_all, evaluate_generalization
from viz.solution_viz import plot_training_curves, compare_routes

evaluate_all(policy, n_instances=256, device="cuda")
evaluate_generalization(policy, sizes=(10, 20, 50), device="cuda")

plot_training_curves(history)
compare_routes(generate_instance(seed=20_000_123, n_customers=20), policy, device="cuda")
```

> Sur Colab, retirer `matplotlib.use("Agg")` des scripts pour un affichage
> en ligne, ou utiliser directement les fonctions comme ci-dessus.

## Paramétrage

Tout est centralisé dans `config.py` et surchargeable à l'appel :

```python
generate_instance(
    seed=0,
    n_customers=50,
    spatial_method="clustered",     # "uniform" | "clustered"
    degree_of_dynamism=0.7,          # part de clients apparaissant en cours de journée
    arrival_distribution="poisson",  # "uniform" | "poisson"
    capacity=40,
)
```

## Structure

| Dossier | Rôle |
|---|---|
| `data/` | Génération et validation d'instances |
| `viz/` | Cartes, tableaux de bord, tournées, courbes |
| `rl/` | Environnement, GNN, politique, entraînement, évaluation |
| `scripts/` | Points d'entrée numérotés |
| `outputs/` | Datasets, figures, points de reprise |

## Résultats de référence

20 clients, 60 époques sur CPU, test sur 256 instances jamais vues :

| Méthode | Distance moyenne | Écart |
|---|---|---|
| Aléatoire | 1177,3 | +59,6 % |
| Plus proche voisin | 737,6 | référence |
| **RL + GNN** | **673,1** | **−8,8 %** |

Généralisation (entraîné sur 20 clients) : −3,4 % sur 10 clients,
−8,2 % sur 50 clients.

Détails et limites dans le cahier des charges.

## Problème dynamique (information en ligne) — version actuelle

La politique ne connaît une requête qu'à partir de son instant d'apparition.

- `DVRPEnv(..., information="online")` (défaut) : `env.observe()` ne contient
  que le dépôt et les requêtes apparues (attributs, distances, comptages) ;
  `information="clairvoyant"` n'est conservé que comme ablation.
- `DVRPPolicy(information=...)` ne lit que `env.observe()` ; le graphe
  observable est réencodé à chaque apparition (seulement pour les instances
  du lot concernées). Le régime est enregistré dans les points de reprise.
- `rl/reopt.py` : heuristique de réoptimisation à chaque événement (OR-Tools).
- Tests : `python -m pytest tests/` (dont `tests/test_dynamic.py`, invariance
  à l'information cachée, événements, parité entraînement/inférence).
- Campagne : `EP=200 python scripts/17_dynamic_campaign.py`, analyse :
  `python scripts/18_analyse_dynamic.py`, latence :
  `python scripts/15_latency_single.py`, sonde de fuite :
  `python scripts/19_leak_probe.py outputs/dynamic/online_s0.pt`.
