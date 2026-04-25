# First Baseline

Décision actuelle de départ:

- benchmark principal: `Crafter`
- benchmark secondaire de contrôle: `DMControl`
- famille de base: `Dreamer-style world model`
- activation candidate principale: `SwiGLU`
- baseline comparaison simple: `GELU`
- `MoE` autorisé dès la v1

## Baseline v1 proposée

### World model

- encodeur visuel compact
- latent séquentiel de type `Dreamer / RSSM`
- transition model compacte
- reconstruction / prediction heads minimales

### Policy

- actor-critic dans l’imagination
- horizon d’imagination modéré au début

### MLP / activations

- version dense baseline:
  - `GELU`
- version principale:
  - `SwiGLU`

### MoE v1

Version volontairement simple:

- `4` experts totaux
- `2` experts actifs
- router `top-k`
- pas de sophistication inutile en v1
- MoE uniquement dans le bloc MLP principal du world model ou du transition block

Pourquoi si petit:
- on veut tester le principe
- pas exploser la complexité
- garder le coût de debug raisonnable

## Hypothèse de recherche

Hypothèse principale:
- sur `Crafter`, un petit `MoE` peut aider car l’environnement contient plusieurs sous-régimes
  - exploration
  - collecte
  - crafting
  - survie
  - navigation

Hypothèse secondaire:
- `SwiGLU` donnera un meilleur compromis que `GELU` dans le bloc MLP

## Ce qu’on veut apprendre en premier

- est-ce que `Crafter` tourne assez vite pour itérer
- est-ce que la baseline dense apprend quelque chose de réel
- est-ce que `SwiGLU` aide déjà sans `MoE`
- est-ce que le `MoE` simple bat une baseline dense à budget actif comparable

## Ce qu’on ne fait pas encore

- pas de gros `MoE`
- pas de hiérarchie complexe d’experts
- pas de token world model compliqué
- pas de long planning sophistiqué
- pas de mélange de 10 idées à la fois
