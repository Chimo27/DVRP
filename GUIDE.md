# Guide d'utilisation — DVRP_RL_GNN

Ce guide décrit comment reproduire l'intégralité des résultats du rapport, et
comment produire ceux qui n'ont pas encore été obtenus.

---

## 1. Installation

### Sur Google Colab (recommandé)

**Activer le GPU d'abord** : `Exécution → Modifier le type d'exécution → GPU`.

```python
!pip install -q torch_geometric ortools
```

```python
from google.colab import files
files.upload()          # déposer DVRP_RL_GNN.zip
!unzip -q DVRP_RL_GNN.zip
%cd DVRP_RL_GNN
```

PyTorch, NumPy, Matplotlib et SciPy sont déjà présents sur Colab.

> Un message d'erreur de dépendances `protobuf` peut apparaître après
> l'installation d'`ortools`. Il provient de bibliothèques Google IA
> préinstallées sur Colab et **n'affecte pas ce projet**. Vérifier simplement
> que les imports passent :
>
> ```python
> import torch, torch_geometric
> from ortools.constraint_solver import pywrapcp
> print("OK |", torch.__version__, "| GPU:", torch.cuda.is_available())
> ```

### En local

```bash
pip install torch torch_geometric numpy matplotlib scipy ortools
```

---

## 2. Sauvegarder les points de reprise (à lire avant de lancer quoi que ce soit)

**Colab efface le disque à chaque nouvelle session.** Les entraînements longs
de ce projet durent plusieurs heures : sans sauvegarde sur Drive, tout est
perdu à la déconnexion, et les scripts d'évaluation qui exigent un checkpoint
échoueront avec `FileNotFoundError`.

Monter Drive **dès le début** de chaque session :

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/dvrp_checkpoints
```

Après chaque entraînement :

```python
!cp outputs/final/*.pt /content/drive/MyDrive/dvrp_checkpoints/
!cp outputs/final/*.json /content/drive/MyDrive/dvrp_checkpoints/
```

Au début d'une session suivante :

```python
!mkdir -p outputs/final
!cp /content/drive/MyDrive/dvrp_checkpoints/*.pt outputs/final/
```

---

## 3. Vérification rapide (2 minutes)

Avant tout calcul long, s'assurer que l'environnement est sain :

```bash
python tests/test_invariants.py
```

Sortie attendue : `24/24 tests passés`. Un échec signale un problème
d'installation ou une régression du code — inutile de lancer un entraînement
tant que ce n'est pas résolu.

---

## 4. Résultats déjà obtenus — comment les reproduire

### 4.1 Campagne principale (~90 min sur GPU)

Produit les trois entraînements dont sont issus tous les résultats des
chapitres 7 et 8 du rapport.

```bash
python scripts/06_final_experiments.py --device cuda --epochs 300
```

| Fichier produit | Contenu |
|---|---|
| `outputs/final/policy_principal.pt` | modèle principal (congestion + objectif temps) |
| `outputs/final/policy_sans_congestion.pt` | ablation sans congestion |
| `outputs/final/policy_domain_rand.pt` | ablation avec randomisation de domaine |
| `outputs/final/resultats_rapport.json` | **tous les résultats en un fichier** |

Pour n'obtenir que le modèle principal (~30 min au lieu de 90) :

```bash
python scripts/06_final_experiments.py --device cuda --epochs 300 --skip-ablations
```

### 4.2 Comparaison au solveur exact (~45 min)

Situe le modèle par rapport à une borne inférieure prouvée. **Exige
`policy_principal.pt`.**

```bash
python scripts/08_compare_exact.py \
    --checkpoint outputs/final/policy_principal.pt \
    --sizes 8 10 12 --n-instances 30 --time-limit 30 --device cuda
```

Produit `outputs/final/comparaison_exacte.json`.

Le temps est dominé par OR-Tools : 30 s × 90 instances ≈ 45 min. Le solveur
consomme tout son budget même après avoir prouvé l'optimum. Pour un essai
rapide : `--sizes 8 --n-instances 5 --time-limit 10` (≈ 1 min).

### 4.3 Coût d'inférence (~5 min)

```bash
python scripts/11_inference_time.py --sizes 10 20 50 100 --device cuda
```

Produit `outputs/final/temps_inference.json`. Ne nécessite **pas** de
checkpoint : le coût d'une passe avant ne dépend pas de la valeur des poids.

### 4.4 Régénérer les figures du rapport

Depuis le dossier du rapport, avec les JSON à côté :

```bash
python generate_figures.py
```

---

## 5. Résultats non encore obtenus

Ces trois expériences sont écrites et testées, mais n'ont jamais été lancées à
pleine échelle. Elles répondent aux limites signalées au chapitre 8.

### 5.1 Ré-évaluation en régime cohérent (~10 min) — priorité haute

Corrige le décalage entraînement/test signalé pour les tableaux 7.5 et 7.7.
Aucun réentraînement : recharge simplement les checkpoints.

```bash
python scripts/07_reevaluate.py --device cuda
```

**Exige les trois checkpoints** dans `outputs/final/`. Produit
`outputs/final/reevaluation.json`.

### 5.2 Répétition multi-graines (~5 h)

Répond à « chaque configuration correspond à un unique entraînement ». Les
graines sont partagées entre configurations, ce qui autorise un test *t*
apparié.

```bash
python scripts/09_multi_seed.py \
    --configs principal sans_congestion \
    --n-seeds 5 --epochs 300 --device cuda
```

Produit `outputs/final/multi_seed/resultats_multi_seed.json`.

**Si le temps manque**, réduire `--n-seeds 3` (≈ 3 h) plutôt que `--epochs` :
des entraînements plus courts ne seraient plus comparables à ceux du rapport.

### 5.3 Sensibilité au degré de dynamisme

Le DoD est le paramètre qui définit le problème, et la campagne actuelle le
maintient constant à 0,5.

**Mode rapide** (~10 min) — robustesse d'une politique existante :

```bash
python scripts/12_dod_sweep.py \
    --mode eval --checkpoint outputs/final/policy_principal.pt \
    --dods 0.0 0.2 0.4 0.6 0.8 1.0 --device cuda
```

**Mode complet** (~3 h) — une politique entraînée par valeur de DoD, chacune
évaluée dans son propre régime. C'est le seul mode permettant de conclure sur
l'effet propre du dynamisme :

```bash
python scripts/12_dod_sweep.py --mode train --epochs 300 --device cuda
```

Produit `outputs/final/dod_sweep.json`.

### 5.4 Ablation baseline figée vs mobile (~3 h)

Valide expérimentalement l'argument analytique de la section 5.4.1.

```bash
python scripts/10_baseline_ablation.py --n-seeds 3 --epochs 300 --device cuda
```

Produit `outputs/final/baseline_ablation/resultats_baseline_ablation.json`.

---

## 6. Ordre recommandé

Si tout est à refaire depuis zéro :

```
1. tests/test_invariants.py          2 min    vérification
2. 06_final_experiments.py          90 min    → 3 checkpoints + résultats
   ↳ sauvegarder les .pt sur Drive
3. 07_reevaluate.py                 10 min    corrige le décalage congestion
4. 11_inference_time.py              5 min    coût d'inférence
5. 08_compare_exact.py              45 min    borne inférieure prouvée
6. 12_dod_sweep.py --mode eval      10 min    sensibilité au dynamisme
   ───────────────────────────────────────
   Total                           ~2 h 45    couvre tout le rapport actuel
                                              + les deux correctifs prioritaires

7. 09_multi_seed.py                  ~5 h     validité statistique
8. 12_dod_sweep.py --mode train      ~3 h     effet propre du dynamisme
9. 10_baseline_ablation.py           ~3 h     ablation baseline
```

Les étapes 1 à 6 tiennent dans une session Colab. Les étapes 7 à 9 demandent
plusieurs sessions — d'où l'importance de la sauvegarde sur Drive.

---

## 7. Usage courant

### Entraîner un modèle avec des paramètres personnalisés

```bash
python scripts/03_train.py \
    --n-customers 30 --epochs 200 --batch-size 128 \
    --device cuda --out outputs/checkpoints/mon_modele.pt
```

### Évaluer un modèle

```bash
python scripts/04_evaluate.py \
    --checkpoint outputs/checkpoints/mon_modele.pt \
    --n-customers 30 --n-instances 256 --device cuda
```

### Résoudre une instance et afficher les tournées

```python
from rl.train import load_checkpoint
from rl.infer import solve_instance, print_solution
from data.generator import generate_instance

policy, _ = load_checkpoint("outputs/final/policy_principal.pt")

# Graine ≥ 20 000 000 : plage réservée au test, jamais vue à l'entraînement
instance = generate_instance(seed=20_000_777, n_customers=20)

tours, distance = solve_instance(policy, instance)
print_solution(tours, instance)
```

Sortie :

```
Véhicule 1 : Dépôt → Client 19 → Client 4 → ... → Dépôt   (distance 232.2)
Véhicule 2 : Dépôt → Client 18 → Client 16 → ... → Dépôt  (distance 179.0)
...
```

### Générer et inspecter une instance

```python
from data.generator import generate_instance
from data.validation import validate_instance, print_validation_report
from viz.instance_viz import plot_instance_dashboard

inst = generate_instance(seed=42, n_customers=20)
print(inst.summary())
print_validation_report(*validate_instance(inst))
plot_instance_dashboard(inst)
```

---

## 8. Paramètres importants

| Paramètre | Valeurs | Rôle |
|---|---|---|
| `congestion_mode` | `off`, `faithful`, `normalized`, `multiplicative` | modèle de temps de trajet (cf. §4.4 du rapport) |
| `objective` | `distance`, `time`, `combined` | grandeur optimisée. **Avec congestion, utiliser `time`** : sinon le modèle subit les heures de pointe sans raison de les éviter |
| `degree_of_dynamism` | 0.0 à 1.0 | proportion de clients révélés en cours de journée |
| `domain_randomization` | booléen | fait varier la configuration d'un lot à l'autre |

Les plages de graines sont disjointes par construction :
entraînement (incrémentale), évaluation (10⁷), test (2×10⁷),
robustesse (3×10⁷), balayage DoD (4×10⁷).

---

## 9. Problèmes courants

**`FileNotFoundError: outputs/final/policy_principal.pt`**
Le checkpoint n'existe pas dans la session courante. Soit le recopier depuis
Drive (§2), soit relancer `06_final_experiments.py`.

**`CUDA out of memory`**
Réduire `--batch-size` (128 → 64 → 32). L'entraînement sera plus lent mais
les résultats restent comparables à batch équivalent.

**Erreur de dépendances `protobuf` après installation d'ortools**
Sans effet sur ce projet (§1).

**Session Colab déconnectée pendant un entraînement long**
`06_final_experiments.py` sauvegarde toutes les 20 époques
(`checkpoint_every=20`) : le dernier point de reprise est récupérable si le
dossier a été synchronisé sur Drive.

**Les résultats diffèrent légèrement d'une exécution à l'autre**
Attendu sur GPU : certaines opérations CUDA ne sont pas déterministes au bit
près. Les écarts doivent rester marginaux ; un écart important signale un
problème.
