# Engram Paper Comparison

Comparaison entre:
- le papier local `Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models`
- les papiers / rapports liés `mHC` et `DSA`
- notre implémentation actuelle dans `src/models/`

Base locale utilisée:
- `.cache/pdftxt/engram.txt`
- `.cache/pdftxt/mhc.txt`
- `.cache/pdftxt/dsa_report.txt`
- `.cache/pdftxt/deepseek_v32.txt`

## Verdict court

On a actuellement:
- une **ossature inspirée** de `Engram + mHC + DSA`
- pas une implémentation fidèle au niveau papier / système / entraînement

Le plus proche du papier aujourd’hui:
- présence d’un module mémoire `EngramMemory`
- présence d’un residual multi-branch simplifié
- présence d’une sparse attention locale+globale simplifiée
- présence d’un MoE sparse

Ce qui manque encore est majeur:
- le vrai chemin `N-gram hashed lookup`
- la vraie compression tokenizer
- la vraie fusion branch-specific dans un backbone `mHC` crédible
- le vrai `DSA` avec indexer entraîné par KL et sélection top-k de KV
- les détails de placement, d’initialisation et de système

## Ce que le papier Engram fait vraiment

### 1. Le cœur d’Engram n’est pas une simple mémoire learnable

Dans le papier, `Engram` n’est pas juste une table mémoire dense interrogée par attention.

Le design central est:
- suffix `N-grams`
- compression tokenizer préalable
- hashing déterministe multi-head
- lookup dans de grandes tables d’embeddings
- gating contextuel avec l’état caché courant
- convolution causale depthwise légère
- insertion seulement à certaines couches

Donc:
- ce n’est pas un mémoire retrieval “à la Perceiver” ou “à la kNN attention”
- c’est une mémoire **statique, discrète, lookup-based, O(1)** par construction

### 2. Configuration principale du papier

Le papier indique notamment:
- `Engram-27B` dérivé d’un `MoE-27B`
- experts routés réduits de `72` à `55`
- budget réalloué vers `5.7B` paramètres de mémoire Engram
- `Engram` inséré aux couches `2` et `15`
- `N-gram max = 3`
- `heads = 8`
- `dimension = 1280`
- optimisation séparée:
  - embeddings avec `Adam`
  - learning rate `5x`
  - pas de weight decay
- convolution initialisée à zéro pour préserver l’identité au départ

### 3. Benchmarks passés dans le papier Engram

Le papier évalue:

- language modeling
  - The Pile loss
  - validation loss interne

- knowledge & reasoning
  - MMLU
  - MMLU-Redux
  - MMLU-Pro
  - CMMLU
  - C-Eval
  - AGIEval
  - ARC-Easy / ARC-Challenge
  - TriviaQA
  - TriviaQA-ZH
  - PopQA
  - CCPM
  - BBH
  - HellaSwag
  - PIQA
  - WinoGrande

- reading comprehension
  - DROP
  - RACE Middle / High
  - C3

- code & math
  - HumanEval
  - MBPP
  - CruxEval
  - GSM8K
  - MGSM
  - MATH

- long context
  - LongPPL
  - RULER

### 4. Gains rapportés par le papier Engram

Le papier insiste sur:
- `MMLU +3.4`
- `CMMLU +4.0`
- `BBH +5.0`
- `ARC-Challenge +3.7`
- `HumanEval +3.0`
- `MATH +2.4`
- `GSM8K +2.2`

Et long-context:
- `Multi-Query NIAH: 84.2 -> 97.0`
- `Variable Tracking: 77.0 -> 89.0`

### 5. Ablations importantes du papier Engram

Le papier montre que les trois éléments les plus importants sont:
- `multi-branch integration`
- `context-aware gating`
- `tokenizer compression`

Autres résultats importants:
- une seule insertion optimale vers `Layer 2`
- la config à deux injections fait mieux que la version mono-couche
- enlever la depthwise conv dégrade peu
- ajouter des `4-grams` n’aide pas forcément sous budget fixe

### 6. Système / infra dans le papier Engram

Le papier a un vrai angle système:
- offload en RAM hôte
- préfetch déterministe grâce aux IDs de lookup
- overhead rapporté < `3%` pour une très grosse table

Donc Engram dans le papier = architecture + système + stratégie de scaling

## Ce que mHC fait vraiment dans le papier

`mHC` n’est pas juste “plusieurs branches résiduelles”.

Le papier ajoute:
- une contrainte de manifold / mélange structuré
- pré / post / residual mappings
- normalisation et paramétrisation spécifiques
- projection structurée avec contraintes type Sinkhorn pour certaines composantes
- très gros travail d’optimisation système
- overhead système rapporté autour de `6.7%`

Le papier montre:
- meilleure stabilité que `HC`
- meilleure loss finale
- gains downstream sur:
  - `BBH`
  - `DROP`
  - `GSM8K`
  - `MATH`
  - `MMLU`
  - `PIQA`
  - `TriviaQA`

Donc notre `MultiBranchResidual` actuel:
- capture vaguement l’idée “plusieurs branches”
- mais ne capture pas le cœur mathématique / contraint / infrastructurel de `mHC`

## Ce que DSA fait vraiment dans le papier

`DSA` n’est pas juste “local attention + quelques tokens globaux”.

Dans les rapports DeepSeek:
- `DSA` est instancié sous `MLA`
- via un mode `MQA` pour efficacité
- il y a un `lightning indexer`
- ce dernier est entraîné avec une perte `KL` contre la distribution de l’attention dense
- warm-up séparé de l’indexer
- puis sparse training complet
- sélection de `2048` key-value tokens par query en stage sparse
- entraînement long contexte `128K`

Donc notre attention actuelle:
- imite une sparse pattern locale + globale
- mais n’implémente pas le mécanisme central du papier, à savoir l’indexer appris et la sélection fine top-k de KV

## Comparaison directe avec notre code

### Engram

Ce qu’on a:
- `EngramMemory`
- mémoire learnable
- retrieval `top-k`
- projection query
- fusion résiduelle

Ce qui manque par rapport au papier:
- `N-gram lookup`
- hashing déterministe
- compression tokenizer
- séparation par ordre de n-gram
- multi-head hashing réel
- vrai design O(1) lookup statique
- gating exactement basé sur hidden/query + memory key/value du papier
- depthwise causal conv Engram
- placement contrôlé à couches spécifiques
- optimisation dédiée embeddings / conv
- offload/prefetch

Conclusion:
- notre `EngramMemory` actuel est **un proxy de mémoire conditionnelle**
- ce n’est **pas encore Engram au sens du papier**

### mHC

Ce qu’on a:
- `MultiBranchResidual`
- mélange pondéré des branches
- projection concaténée

Ce qui manque:
- vraie structure `pre/post/res`
- contraintes manifold / stochasticité
- design mathématique du papier
- kernels fusionnés / infra
- logique de stabilité spécifique

Conclusion:
- on a une **approximation conceptuelle**
- pas une implémentation `mHC`

### DSA

Ce qu’on a:
- masquage local + accès périodique à des positions globales

Ce qui manque:
- `lightning indexer`
- warm-up dense avec KL
- sélection top-k de key-values
- instanciation sous `MLA`
- vraie logique de sparse KV retrieval

Conclusion:
- on a une **sparse attention simplifiée**
- pas `DSA`

## État de fidélité global

Si on note la fidélité au papier:

- `Engram`: faible à moyenne
- `mHC`: faible
- `DSA`: faible

Si on note la valeur comme base de recherche:

- `Engram`: bonne
- `mHC`: correcte
- `DSA`: correcte

Autrement dit:
- comme “prototype research-friendly”, la base actuelle est utile
- comme “implémentation fidèle au papier”, elle est insuffisante

## Ce qu’il faut faire pour une vraie évaluation honnête

Avant de parler de “passer les mêmes tests que le papier”, il faut au minimum:

1. Refaire `Engram` comme lookup mémoire statique
- tokenizer compression
- 2/3-gram hashing
- tables mémoire séparées
- gating contextuel
- depthwise conv

2. Refaire `DSA` comme vraie sparse KV attention
- indexer entraîné
- top-k KV per query
- protocole dense warm-up + sparse training

3. Refaire `mHC` de façon plus fidèle
- structure pré/post/res
- mélange contraint
- au moins une version mathématiquement plus proche

4. Implémenter le protocole d’ablation du papier
- insertion layer sweep
- w/o tokenizer compression
- w/o multi-branch
- w/o context-aware gating
- w/o conv
- 2/3-gram vs 4-gram

## Conclusion pratique

On ne doit pas dire actuellement:
- “on a implémenté Engram, mHC et DSA”

On peut dire honnêtement:
- “on a implémenté un transformer de recherche avec des **approximations inspirées** de Engram, mHC et DSA”

La bonne suite n’est pas d’évaluer tout de suite contre les benchmarks du papier.
La bonne suite est:

1. choisir la première brique à rendre fidèle
2. rendre cette brique vraiment proche du papier
3. seulement ensuite lancer une évaluation sérieuse

## Priorité recommandée

Ordre conseillé:

1. `Engram`
   - c’est la brique la plus distinctive
   - le papier donne un chemin clair

2. `DSA`
   - utile si on veut du vrai long-context sparse

3. `mHC`
   - plus subtil à refaire fidèlement
   - probablement après
