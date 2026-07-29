# World Models / JEPA — analyse du transcript de l’épisode Yann LeCun

Source : [À la French — Les World Models : l’IA post LLM, expliqué par Yann LeCun](https://www.youtube.com/watch?v=m7ywFu3Yqh8), publié le 7 juillet 2026, durée 1 h 32 min.

Fichiers archivés :

- `youtube/A la french/m7ywFu3Yqh8_Les Worlds Models : l’IA post LLM, expliqué par Yann LeCun/`
- transcript automatique français (`fr.vtt`, `fr-orig.vtt`), sous-titres anglais (`en-en.vtt`), description et métadonnées JSON.

Le transcript est une source d’idées et d’affirmations, pas une preuve scientifique. Les formulations ci-dessous sont paraphrasées afin de ne pas reproduire inutilement le contenu protégé.

## Chronologie des idées

| Moment | Idée extraite | Statut initial |
|---|---|---|
| 20:50–23:30 | Un world model prédit les conséquences d’actions et peut être appris de façon auto-supervisée. | Définition compatible avec la littérature ; validation empirique nécessaire. |
| 23:30–27:30 | Les détails pixel par pixel sont souvent imprédictibles et une représentation abstraite serait préférable. | Hypothèse plausible ; les pixels restent parfois utiles. |
| 27:30–34:00 | Les animaux apprennent une physique intuitive par expérience et abstraction, sans résoudre explicitement les équations. | Analogie utile, pas équivalence démontrée avec un réseau. |
| 38:00–46:30 | La prédiction latente, plutôt que la génération de tokens/pixels, est proposée pour construire une représentation dynamique. | Soutenue par JEPA/V-JEPA ; portée générale encore ouverte. |
| 47:30–55:30 | Le collapse est le problème central des objectifs de prédiction d’embeddings ; les solutions discutées incluent Barlow Twins, VICReg, BYOL et JEPA. | Problème bien documenté ; les mécanismes de prévention diffèrent. |
| 56:00–68:00 | Une régularisation vers une distribution gaussienne isotrope est présentée comme une cible géométrique pratique. | Fondée sur LeJEPA/SIGReg ; ce n’est pas une loi physique universelle. |
| 68:00–76:30 | Les world models doivent capturer des abstractions suffisantes pour planifier, plutôt que simuler tous les détails. | Compatible avec les travaux de planification latente ; dépend de la tâche. |
| 80:00–87:00 | Tapestry est présenté comme une fédération d’acteurs visant une IA ouverte, diverse et souveraine. | Proposition d’organisation/infrastructure, distincte de la qualité d’un modèle. |

## Confrontation scientifique

### 1. « Le prochain token ne suffit pas à comprendre le monde »

À distinguer : un modèle de langage peut apprendre des régularités, des faits et
des structures latentes à partir du texte ; cela ne prouve pas qu’il possède un
modèle causal, perceptif et actionnable du monde physique. La formulation
« ne pourra jamais » est trop forte sans définition opérationnelle de
« comprendre ».

Les travaux JEPA déplacent la cible vers la prédiction de représentations
abstraites. I-JEPA prédit des blocs d’image dans l’espace d’embedding plutôt
que des pixels [papier I-JEPA](https://arxiv.org/abs/2301.08243). V-JEPA apprend
des prédictions vidéo sans reconstruction pixel et V-JEPA 2 ajoute
compréhension, anticipation et planification avec des données vidéo et robotique
[​V-JEPA](https://arxiv.org/abs/2404.08471), [V-JEPA 2](https://arxiv.org/abs/2506.09985).

### 2. « Un world model prédit l’effet d’une action »

C’est la formulation la plus testable : apprendre `z_{t+1} = f(z_t, a_t)` puis
utiliser `f` pour sélectionner des actions. Le papier historique *World Models*
montre déjà une représentation spatio-temporelle compressée utilisable par une
politique, y compris dans un environnement simulé par le modèle
[​Ha & Schmidhuber](https://arxiv.org/abs/1803.10122). Les travaux récents
V-JEPA 2-AC et JEPA-WM fournissent des validations robotiques plus proches de
cette définition.

Dans notre dépôt, cette boucle n’existe pas encore : nous avons des probes de
rappel, du LM, LeJEPA textuel et des briques RL/OCR, mais pas encore un
encodeur d’observation + prédicteur latent conditionné par action + planificateur
évalué dans Gymnasium.

### 3. « Il faut prédire dans un espace latent plutôt que générer des pixels »

L’argument est computationnel : les détails visuels imprédictibles peuvent
être traités comme du bruit pour une tâche de contrôle. Mais les pixels peuvent
rester nécessaires pour la fidélité visuelle, la perception fine ou certaines
actions. La bonne question expérimentale est donc le rapport coût/planification
à qualité de tâche, pas une opposition absolue latent/pixel.

### 4. Collapse et régularisation

Barlow Twins rapproche les vues positives et réduit la redondance via une
matrice de corrélation proche de l’identité [papier](https://arxiv.org/abs/2103.03230).
VICReg sépare explicitement invariance, variance et covariance
[​papier](https://arxiv.org/abs/2105.04906). LeJEPA propose SIGReg et l’hypothèse
de l’embedding gaussien isotrope [papier LeJEPA](https://arxiv.org/abs/2511.08544).

Cela correspond directement à notre Step 2 : la LeJEPA réelle est implémentée,
mais nos essais text-byte n’ont pas amélioré `engram_noconv`. Ce résultat ne
réfute pas JEPA pour la vidéo ou le contrôle ; il dit seulement que l’auxiliaire
testé n’a pas aidé ce proxy LM.

### 5. « Pas besoin de tokenizer »

La vidéo défend des séquences de représentations issues de fenêtres vidéo,
mais cela ne signifie pas que toute architecture JEPA est sans patching,
discrétisation ou tokenizer. I-JEPA et V-JEPA utilisent des unités de contexte
adaptées au signal. Dans notre dépôt, `byte` et `symbolic` sont des modes
distincts ; aucun résultat actuel ne justifie de généraliser « sans tokenizer »
à tous les LLM.

### 6. Abstraction et intuition physique

L’idée d’une hiérarchie d’abstractions est compatible avec les représentations
multi-échelles, mais l’analogie chat/rat ≠ preuve de mécanisme. Il faut tester
des invariances et des prédictions contrefactuelles : collisions, gravité,
occlusion, changement de caméra, intervention et action non observée.

### 7. Tapestry et souveraineté

La fédération des données/paramètres peut traiter gouvernance, diversité et
localité des données. Elle ne garantit ni convergence ni absence de biais. Ces
questions doivent être séparées des expériences d’architecture du modèle.

## Ce que nos expériences établissent déjà

- Le rappel `variable_tracking` était mal évalué au départ : l’ancien script
  était évaluation-only et ne faisait pas d’apprentissage. Le pipeline entraîné
  apprend la tâche à longueur 64 (~99 %) et devient sensible au seed à 256.
- Sur la tâche entraînée à longueur 256, GELU² est le plus stable (93,36 %),
  tandis que ReLU² transfère mieux vers passkey/multi-query.
- Le schedule probabiliste à branche unique est plus rapide que la moyenne
  pondérée et a un compromis intéressant, mais ne constitue pas encore une
  preuve de gain LLM.
- LeJePA textuel et les probes actuelles ne testent pas un world model
  action-conditionné.

## Programme expérimental dérivé

1. Construire un environnement Gymnasium minimal avec observations visuelles ou
   symboliques, actions et transitions contrôlées.
2. Comparer un modèle pixel/autoencodeur, un prédicteur latent JEPA et un
   prédicteur latent action-conditionné.
3. Mesurer erreur de prédiction, qualité de planification, robustesse aux
   interventions et coût énergie/FLOPs.
4. Ajouter SIGReg/LeJEPA, Barlow Twins et VICReg comme ablations séparées ;
   surveiller collapse, covariance, variance et mutual information.
5. Tester le transfert : entraînement sur une distribution, évaluation sur
   une autre dynamique, caméra ou composition d’objets.
6. Comparer ensuite les activations dynamiques, Engram, AttnRes/mHC et la
   récurrence avec un budget de calcul égal.
7. Intégrer la branche [JEPA + Simulus](../papers/targets/jepa_simulus/README.md) :
   flux latents modulaires, prédiction du prochain état latent conditionnée par
   l'action, puis planification/politique. La branche Simulus est une inspiration
   architecturale et non une implémentation déjà validée dans ce dépôt.

## Références historiques à conserver

AlexNet est un rappel méthodologique important : le gain de 2012 venait d’une
combinaison architecture + données + GPU + régularisation, pas d’une seule
activation. Voir [Krizhevsky, Sutskever & Hinton, NeurIPS 2012](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).
Pour la quantification, les résultats AlexNet/ResNet montrent aussi qu’il faut
rapporter la perte de qualité à plusieurs bit-widths et pas seulement annoncer
« 1 bit » [Learned Quantization](https://arxiv.org/abs/1808.05779).
