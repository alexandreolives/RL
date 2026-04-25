# Gemma 4 MoE And Activations

Notes rapides sur `Gemma 4`, surtout la variante `26B-A4B`, avec un focus sur le MoE et les fonctions d'activation.

## Sources primaires utilisées

- Blog Google Gemma 4:
  `https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/`
- Model card officielle Hugging Face:
  `https://huggingface.co/google/gemma-4-26B-A4B`
- Config officielle du modèle:
  `https://huggingface.co/google/gemma-4-26B-A4B/blob/main/config.json`

## Ce qui est sûr sur le MoE de Gemma 4

Pour `google/gemma-4-26B-A4B`:

- paramètres totaux: `25.2B`
- paramètres actifs par token: `3.8B`
- layers: `30`
- experts: `128 total`
- experts actifs: `8`
- expert partagé: `1 shared`

Le point important est donc:
- Gemma 4 `26B-A4B` n'active pas tout son FFN dense à chaque token
- le routeur sélectionne `8` experts parmi `128`
- il existe aussi `1` expert partagé qui semble toujours présent dans le bloc MoE

Interprétation pratique:
- grosse capacité mémoire grâce au grand pool d'experts
- coût actif proche d'un petit modèle
- bon compromis si on cherche de la qualité sans payer tout le coût d'un dense 25B+

## Ce qui est documenté sur l'architecture autour du MoE

La model card indique aussi:

- attention hybride:
  - sliding window local attention
  - couches globales intercalées
  - dernière couche toujours globale
- `attention_k_eq_v: true` sur les couches globales
- usage de `p-RoPE`
- contexte jusqu'à `256K`

Donc l'efficacité de Gemma 4 ne vient pas seulement du MoE:
- il y a aussi une stratégie d'attention pensée pour limiter le coût mémoire sur long contexte

## Fonction d'activation utilisée par Gemma 4

La config publique du modèle indique explicitement:

- `hidden_activation: "gelu_pytorch_tanh"`

Donc, pour la variante publique Gemma 4 `26B-A4B`, l'activation déclarée dans la config est:

- `GELU` approx version `gelu_pytorch_tanh`

Point important:
- ce n'est pas `SwiGLU`
- ce n'est pas `GeGLU`
- ce n'est pas `ReLU`

Ça veut dire que Google n'a pas choisi ici la mode "GLU partout" pour cette variante exposée publiquement, au moins au niveau du champ d'activation principal de la config.

## Ce qu'on peut raisonnablement inférer sur leur MoE

Ce qui suit est une inférence prudente, pas une phrase explicitement écrite par Google:

- le routeur est très probablement de type `top-k`, avec `k = 8`
- l'expert partagé sert probablement de canal généraliste stable
- les experts routés servent à la spécialisation
- la structure cherche à maintenir une bonne latence sans sacrifier la capacité totale

Ce que je n'ai pas trouvé documenté publiquement, au moins dans les sources officielles facilement accessibles:

- la loss exacte de balancing du router
- la capacité par expert
- s'il y a token dropping sous surcharge
- la structure exacte de l'expert partagé
- à quelles couches précises le MoE apparaît
- les détails fins de l'entraînement du routeur

Donc on connaît bien le "shape" général du MoE, mais pas encore tous les détails de l'implémentation.

## À retenir si ton objectif est perf maximale sous budget

Gemma 4 MoE montre un pattern intéressant:

- beaucoup d'experts en réserve
- peu d'experts réellement actifs
- un expert partagé pour stabiliser
- coût actif qui reste dans la zone d'un petit modèle

Pour un projet `<= 40GB`, ce pattern est très intéressant, surtout si tu veux:
- garder de la capacité totale élevée
- éviter un dense trop gros
- conserver une bonne vitesse d'inférence

## Fonctions d'activation: quoi regarder en pratique

### 1. Le meilleur compromis pratique aujourd'hui: `SwiGLU`

Si on parle de LLMs modernes en pratique, la réponse la plus robuste est:

- `SwiGLU` est souvent le meilleur compromis qualité / stabilité / coût

Pourquoi:
- très bon comportement empirique
- largement adopté dans les architectures modernes
- meilleure expressivité qu'un MLP GELU simple
- coût supplémentaire raisonnable

Si ton objectif est "forte perf sans faire exploser le compute", c'est la première activation à considérer.

### 2. Ce que Gemma 4 utilise publiquement: `GELU`

Gemma 4 `26B-A4B` expose:

- `gelu_pytorch_tanh`

Donc Google a choisi une activation plus simple côté config publique, malgré un modèle très ambitieux architecturalement.

Lecture possible:
- le vrai gain principal est ailleurs: MoE + attention + entraînement + recipe globale
- ils n'ont pas besoin d'une activation exotique pour rendre le modèle très compétitif

### 3. Si tu pensais à une activation "très forte mais plus chiante en compute"

Le nom que tu cherches est possiblement l'une de celles-ci:

- `SoLU` (`Softmax Linear Units`)
- `Maxout`

#### `SoLU`

Pourquoi ça colle:
- activation intéressante et plus "riche" que GELU/ReLU
- pousse la compétition entre neurones via un softmax local
- souvent discutée pour ses propriétés d'interprétabilité et de sélectivité

Pourquoi c'est plus chiant:
- plus coûteuse qu'une activation standard
- moins standard en prod
- pipeline / tuning moins banal que GELU ou SwiGLU

Quand y penser:
- si tu veux explorer des représentations plus sélectives / plus interprétables
- pas forcément le meilleur choix si ton but principal est juste la meilleure perf pratique sous budget serré

#### `Maxout`

Pourquoi ça colle aussi:
- très expressive
- historiquement réputée forte
- peut approximer des fonctions piecewise-linear très riches

Pourquoi c'est plus chiant:
- plus de paramètres
- plus de compute
- plus lourd à déployer proprement dans des stacks LLM modernes

Quand y penser:
- si tu veux maximiser l'expressivité théorique
- rarement le meilleur choix pratique pour un LLM efficient sous contrainte mémoire

## Verdict simple

Si tu veux une hiérarchie utile:

1. `SwiGLU`
   meilleur compromis pratique pour LLM moderne

2. `GELU`
   simple, stable, crédible, encore très viable

3. `SoLU`
   très intéressante, plus exotique, plus coûteuse / moins standard

4. `Maxout`
   expressive, mais en général trop lourde pour être le choix pragmatique

## Ce que je ferais pour notre objectif

Pour un système `RL / world model / compute <= 40GB`:

- en baseline pragmatique:
  - `SwiGLU`
- si on cherche à reproduire une vibe Gemma-like sobre:
  - `GELU`
- si on veut explorer une piste plus expérimentale:
  - `SoLU`, mais seulement dans une branche de recherche, pas comme choix principal

## Résumé ultra-court

- `Gemma 4 26B-A4B` utilise un `MoE`
- `128` experts totaux
- `8` experts actifs par token
- `1` expert partagé
- `3.8B` paramètres actifs pour `25.2B` totaux
- activation publique déclarée: `gelu_pytorch_tanh`
- meilleure activation pratique moderne: `SwiGLU`
- activation probablement "plus optimale mais trop chiante en compute" selon ton souvenir:
  probablement `SoLU`, sinon `Maxout`
