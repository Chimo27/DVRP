# Cahier des charges

**Optimisation Intelligente des Tournées Logistiques en Temps Réel par Deep Reinforcement Learning et Graph Neural Networks (GNN)**

---

## 1. Contexte et problématique

La planification de tournées de livraison est traditionnellement résolue en
mode **statique** : toutes les commandes sont connues le matin, on calcule
un plan, on l'exécute. La réalité de la logistique urbaine est différente —
les commandes arrivent **en continu pendant la journée**, et le plan doit
être révisé en permanence.

Les méthodes exactes (programmation linéaire en nombres entiers) donnent
l'optimum mais ne passent pas à l'échelle en temps contraint. Les
métaheuristiques donnent de bonnes solutions mais doivent être relancées
intégralement à chaque nouvel événement.

L'approche retenue — **apprentissage par renforcement profond couplé à un
réseau de neurones sur graphes** — déplace le coût de calcul vers une phase
d'entraînement hors ligne. En exploitation, la décision est une simple
inférence : quelques millisecondes, compatible avec une prise de décision
en temps réel.

## 2. Problème traité : DVRP (Dynamic Capacitated VRP)

### 2.1 Définition

Soit un dépôt et *n* clients répartis sur une carte. Chaque client *i* a :

- une position `(x_i, y_i)`
- une demande `q_i` (quantité à livrer)
- une durée de service `s_i`
- un **instant de révélation `r_i`** : `r_i = 0` si le client est connu dès
  le départ, `r_i > 0` s'il apparaît en cours de journée

Un véhicule de capacité `Q` part du dépôt, sert les clients, revient au
dépôt pour recharger, et recommence. Le véhicule **ne peut pas se rendre
chez un client avant `r_i`** : cette contrainte est le cœur du caractère
« temps réel » du problème.

**Objectif** : minimiser la distance totale parcourue, tous les clients
étant servis.

### 2.2 Périmètre : ce qui est inclus et ce qui ne l'est pas

| Inclus | Exclu (extensions futures) |
|---|---|
| Capacité véhicule + recharges au dépôt | Fenêtres de temps (VRPTW) |
| Arrivées dynamiques de clients | Flotte hétérogène |
| Distances euclidiennes | Multi-dépôts |
| Durées de service | Temps de trajet stochastiques |

**Justification de l'exclusion des fenêtres de temps.** Une version
antérieure du travail incluait des fenêtres de temps. Elles rendaient une
grande partie des instances partiellement infaisables : lors des essais,
plus de la moitié des clients ne pouvaient pas être servis dans les délais,
et la récompense était dominée par les pénalités de clients manqués plutôt
que par la qualité du routage. Le signal d'apprentissage portait donc sur
« combien de clients sacrifier » plutôt que sur « comment router
efficacement ».

Sans fenêtres de temps, l'objectif devient univoque — **minimiser la
distance** — et l'épisode ne peut se terminer qu'une fois **tous** les
clients servis. Le modèle ne peut plus « gagner » en ignorant des clients.
Le champ `time_windows` reste présent dans la structure de données pour
permettre l'extension ultérieure sans rupture de format.

## 3. Exigences fonctionnelles

### EF-1 — Génération d'instances

- Générer des instances DVRP paramétrables : nombre de clients, capacité,
  distribution spatiale, degré de dynamisme, horizon
- Reproductibilité **totale** à partir d'une graine unique (`seed`)
- Sauvegarde / rechargement de datasets (format `.npz`)
- Traçabilité : chaque instance conserve dans ses métadonnées l'intégralité
  des paramètres ayant servi à la produire

### EF-2 — Modélisation des sources d'incertitude

| Composante | Loi retenue | Justification |
|---|---|---|
| Position clients | Uniforme **ou** clusters gaussiens | L'uniforme est la convention des benchmarks CVRP ; les clusters approchent des quartiers réels |
| Demande | Uniforme discrète `[1, 9]` | Convention dominante en RL-VRP ; contrôle direct du ratio demande/capacité |
| Arrivées dynamiques | Poisson homogène **ou** uniforme | Poisson modélise des commandes tombant au fil de l'eau |
| Durée de service | Uniforme discrète `[5, 15]` min | Variabilité simple, non critique pour l'objectif |

Ces choix sont **paramétrables**, pas figés : le code permet de les
substituer pour comparer des variantes.

### EF-3 — Validation

- Détecter les incohérences : demande supérieure à la capacité, apparition
  hors horizon, matrice de distances non symétrique, etc.
- La validation **rapporte** les violations, elle ne corrige jamais
  silencieusement l'instance
- Validation applicable à un dataset entier

### EF-4 — Visualisation

- Carte d'une instance distinguant clients statiques / dynamiques / dépôt
- Tableau de bord : demandes, arrivées dynamiques, distances au dépôt
- Comparaison des méthodes de génération avec statistiques descriptives
- Tracé des tournées produites, une couleur par tournée successive
- Courbes d'entraînement

### EF-5 — Environnement d'apprentissage

- Environnement **vectorisé** (traitement d'un lot d'instances en parallèle)
- Masquage des actions garantissant qu'aucune action invalide n'est
  sélectionnable
- **Terminaison garantie** de tout épisode
- Métriques séparées : distance, temps total, attente, clients non servis

### EF-6 — Modèle

- Encodeur **GNN** avec les distances comme caractéristiques d'arêtes
- Décodeur auto-régressif de type pointeur
- Généralisation à des tailles d'instances non vues à l'entraînement

### EF-7 — Entraînement et évaluation

- Entraînement par renforcement, sans solutions optimales de référence
- Comparaison systématique à des heuristiques de référence
- Jeux d'entraînement et de test **disjoints** (graines réservées)
- Sauvegarde périodique de points de reprise

## 4. Exigences non fonctionnelles

| Code | Exigence |
|---|---|
| ENF-1 | **Reproductibilité** : une graine identique produit un résultat identique. Aucun appel à `np.random.*` global dans le code de génération |
| ENF-2 | **Modularité** : chaque composante isolée dans un module dédié, substituable sans toucher au reste |
| ENF-3 | **Séparation des responsabilités** : générer / valider / visualiser / entraîner sont quatre couches distinctes ; un générateur n'affiche rien, un validateur ne modifie rien |
| ENF-4 | **Traçabilité** : tout résultat rattachable aux paramètres qui l'ont produit |
| ENF-5 | **Portabilité** : exécutable sur CPU comme sur GPU, en script comme en notebook (Colab) |

## 5. Architecture

```
DVRP_RL_GNN/
│
├── config.py                    Paramètres centralisés
│
├── data/                        GÉNÉRATION
│   ├── instance.py              Structure de données Instance
│   ├── components.py            Briques : spatial, demande, arrivées, service
│   ├── generator.py             Orchestrateur + datasets
│   └── validation.py            Détection d'incohérences
│
├── viz/                         VISUALISATION
│   ├── instance_viz.py          Cartes, tableaux de bord, comparaisons
│   └── solution_viz.py          Tournées, courbes d'entraînement
│
├── rl/                          APPRENTISSAGE
│   ├── env.py                   Environnement vectorisé
│   ├── gnn_encoder.py           Encodeur GNN
│   ├── policy.py                GNN + décodeur pointeur
│   ├── baselines.py             Heuristiques de référence
│   ├── train.py                 Boucle d'entraînement
│   └── evaluate.py              Évaluation comparative
│
├── scripts/                     EXÉCUTION
│   ├── 01_generate_dataset.py
│   ├── 02_visualize_dataset.py
│   ├── 03_train.py
│   └── 04_evaluate.py
│
└── outputs/                     Datasets, figures, points de reprise
```

**Chaîne de traitement**

```
config → générateur → Instance → validation
                          ↓
                    visualisation
                          ↓
              environnement RL → politique GNN → entraînement
                                                      ↓
                                          évaluation vs heuristiques
```

## 6. Choix techniques et justifications

### 6.1 Pourquoi un GNN et non un Transformer

Un problème de tournées est un **graphe** : des nœuds (clients) reliés par
des arêtes valuées (distances). L'auto-attention classique ne voit que les
coordonnées des nœuds et doit ré-apprendre indirectement la notion de
distance.

Le modèle utilise `GATv2Conv` (PyTorch Geometric) avec `edge_dim=1` : la
distance entre deux nœuds est injectée comme **caractéristique d'arête**
dans le calcul du poids d'attention. L'information structurelle centrale du
problème est ainsi donnée explicitement au réseau.

Bénéfice mesurable : le modèle **généralise à des tailles d'instances non
vues** (cf. §8), puisqu'il apprend une règle de décision locale sur un
graphe plutôt qu'une fonction sur un vecteur de taille fixe.

### 6.2 Pourquoi REINFORCE avec une baseline figée

Un premier prototype utilisait une baseline recalculée à chaque lot avec la
politique **courante**. À mesure que la politique devient déterministe, le
tirage stochastique se rapproche du tirage glouton, l'avantage tend vers
zéro et l'apprentissage se fige — comportement effectivement observé
(progression forte sur les 100 premières époques, puis quasi-plateau).

La baseline est désormais une **copie gelée** de la politique, remplacée
seulement lorsque la politique courante est **statistiquement** meilleure
sur un jeu d'évaluation fixe (test *t* apparié, p < 0,05). L'avantage
conserve ainsi un signal exploitable tout au long de l'entraînement.

### 6.3 Encodage unique par épisode

L'encodage GNN est calculé **une fois** par épisode. Les caractéristiques
statiques (position, demande, instant d'apparition) ne changent pas ; ce
qui évolue (clients servis, capacité, temps) passe par le contexte du
décodeur et le masquage. Cela évite un ré-encodage en O(N²) à chaque pas.

Conséquence assumée : le modèle « voit » les clients dynamiques dès le
départ mais ne peut pas les sélectionner avant leur révélation. Il peut
donc **anticiper** les arrivées futures — comportement recherché en temps
réel.

## 7. Protocole expérimental

| Élément | Valeur |
|---|---|
| Instances d'entraînement | Générées à la volée, graines croissantes |
| Jeu d'évaluation | 128–256 instances, graines réservées (`10 000 000+`) |
| Jeu de test | 256 instances, graines réservées (`20 000 000+`) |
| Références | Aléatoire (borne haute), plus proche voisin (référence à battre) |
| Métrique principale | Distance totale moyenne |
| Métrique de généralisation | Écart vs plus proche voisin sur 10 / 20 / 50 clients |

Les trois jeux sont **disjoints par construction** : aucune instance
d'évaluation ou de test ne peut apparaître à l'entraînement.

## 8. Résultats de référence obtenus

Configuration : 20 clients, capacité 30, DoD 0,5, arrivées Poisson,
60 époques sur CPU (~4 s/époque).

**Test sur 256 instances jamais vues :**

| Méthode | Distance moyenne | Clients non servis | Écart vs référence |
|---|---|---|---|
| Aléatoire | 1177,3 | 0 | +59,6 % |
| Plus proche voisin | 737,6 | 0 | référence |
| **RL + GNN (glouton)** | **673,1** | 0 | **−8,8 %** |

**Généralisation** (modèle entraîné sur 20 clients uniquement) :

| Taille | RL+GNN | Plus proche voisin | Écart |
|---|---|---|---|
| 10 clients | 436,2 | 451,8 | −3,4 % |
| 20 clients | 681,8 | 744,7 | −8,4 % |
| 50 clients | 1329,6 | 1447,8 | −8,2 % |

Le modèle conserve son avantage sur des instances **2,5 fois plus grandes**
que celles vues à l'entraînement, ce qui valide l'intérêt du GNN.

**Ces chiffres constituent une base de référence, pas un résultat final** :
l'entraînement est court (60 époques, CPU, lots de 64). Un entraînement sur
GPU, plus long et sur des lots plus grands, devrait améliorer l'écart.

## 9. Limites connues

1. **Comparaison à l'optimum absente.** Le plus proche voisin est une
   heuristique faible. Une comparaison à un solveur exact (petites
   instances) ou à une métaheuristique établie donnerait une mesure de
   qualité plus rigoureuse.
2. **Graphe complet en O(N²).** Au-delà de quelques centaines de clients, un
   graphe creux (k plus proches voisins) deviendrait nécessaire.
3. **Décodeur simplifié** : pointeur à une tête, sans le mécanisme
   multi-tête des architectures de référence.
4. **Véhicule unique à recharges.** Convention standard, mais ne modélise
   pas plusieurs véhicules opérant simultanément.
5. **Paramètres non calibrés sur données réelles.** Les lois retenues sont
   des conventions de la littérature, non ajustées sur un jeu de données
   terrain.

## 10. Feuille de route

**Court terme**
- Entraînement long sur GPU (500+ époques)
- Comparaison à un solveur exact sur petites instances
- Étude de sensibilité au degré de dynamisme

**Moyen terme**
- Réintroduction des fenêtres de temps, une fois la base solide
- Flotte de plusieurs véhicules
- Graphe creux pour le passage à l'échelle

**Long terme**
- Temps de trajet variables selon l'heure
- Calibration sur données logistiques réelles
