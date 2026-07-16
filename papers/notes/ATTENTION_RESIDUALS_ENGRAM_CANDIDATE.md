# Attention Residuals — candidat de combinaison avec Engram

## Source

- vidéo Bycloud : [An Insanely Elegant LLM Architecture Breakthrough Just
  Dropped](https://www.youtube.com/watch?v=iw1VF8HOCrk), publiée le 18 mai 2026
- papier principal : [Attention Residuals](https://arxiv.org/abs/2603.15031)
  (Kimi Team / Moonshot AI)
- code officiel : [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- comparaison discutée : [mHC: Manifold-Constrained
  Hyper-Connections](https://arxiv.org/abs/2512.24880)

Artefacts locaux :

- PDF Attention Residuals :
  `papers/bycloud/20260518_iw1VF8HOCrk_An_Insanely_Elegant_LLM_Architecture_Breakthrough_Just_Dropped__2603.15031.pdf`
- PDF mHC déjà archivé :
  `papers/bycloud/20260217_Gb0jLRC8uWM_DeepSeek_Just_Added_Parameters_Where_There_Were_None__2512.24880.pdf`
- métadonnées, description et sous-titres :
  `youtube/bycloud/iw1VF8HOCrk_An Insanely Elegant LLM Architecture Breakthrough Just Dropped/`

## Idée principale

Avec PreNorm, le résiduel standard additionne les sorties de toutes les couches
avec un poids fixe de un. Cette accumulation peut faire croître la norme de
l'état caché avec la profondeur et diluer la contribution relative de chaque
couche.

Attention Residuals (`AttnRes`) remplace cette somme fixe par une attention
softmax, apprise et dépendante de l'entrée, sur les représentations produites
par les couches précédentes. Chaque couche peut ainsi sélectionner les états
antérieurs dont elle a besoin.

La variante `Block AttnRes` regroupe les couches et conserve seulement un résumé
par bloc. Elle réduit le stockage et le calcul de `O(L)`/`O(L²)` à
`O(N)`/`O(N²)`, où `N` est le nombre de blocs. Le papier indique qu'environ huit
blocs préservent l'essentiel du gain sur les échelles testées.

## Résultats rapportés

- les courbes de scaling d'AttnRes complet et par blocs restent sous le baseline
  à budget de calcul comparable ;
- Block AttnRes atteint la perte d'un baseline utilisant environ `1.25x` plus
  de calcul ;
- l'intégration finale est testée dans Kimi Linear, avec `48B` paramètres au
  total, `3B` activés et `1.4T` tokens de pré-entraînement ;
- le papier rapporte moins de `4%` d'overhead d'entraînement avec pipeline
  parallelism et moins de `2%` de latence supplémentaire en inférence dans les
  conditions décrites ;
- Full AttnRes dépasse `mHC-lite` dans la comparaison publiée, tandis que Block
  AttnRes l'égale avec moins d'I/O mémoire par couche.

Ces résultats proviennent du papier Kimi et ne sont pas encore reproduits dans
ce dépôt.

## État de l'implémentation locale

Full AttnRes est maintenant disponible dans le backbone natif. Chaque attention,
module Engram et MLP produit une source distincte; la sous-couche suivante
agrège toutes les sources antérieures avec sa propre pseudo-requête. Une
agrégation finale précède également la normalisation de sortie.

Variantes contrôlées :

- `baseline` ;
- `attnres` ;
- `engram_noconv` ;
- `engram_noconv_attnres`.

Les pseudo-requêtes sont initialisées directement à zéro, sans consommation du
générateur aléatoire. Les poids partagés sont donc strictement identiques entre
les bras appariés pour une même seed. Le test unitaire vérifie également que les
gradients atteignent à la fois les pseudo-requêtes et les embeddings Engram.

La campagne multi-seed reproductible est décrite dans
`experiments/step4_engram_attnres/README.md`.

La première campagne est terminée. AttnRes seul améliore le baseline, mais la
combinaison où Engram devient une source softmax indépendante est antagoniste.
Résultats complets :
`experiments/step4_engram_attnres/notes/ATTNRES_RESULTS_2026-07-16.md`.

Le correctif fusionné v1 réduit la régression LM sans la supprimer et n'apporte
pas de gain aval mesurable. Résultats :
`experiments/step4_engram_attnres/notes/ATTNRES_V1_RESULTS_2026-07-16.md`.

Le bypass gated v2 reste lui aussi moins bon qu'Engram sur le test LM. La phase
aval a donc été arrêtée selon le critère prévu. Résultats :
`experiments/step4_engram_attnres/notes/ATTNRES_V2_RESULTS_2026-07-16.md`.

Le v3 à gate bornée améliore v2 mais reste moins bon qu'Engram sur les 9 seeds.
La famille du bypass cumulatif est donc rejetée sous ce protocole :
`experiments/step4_engram_attnres/notes/ATTNRES_V3_RESULTS_2026-07-16.md`.

## Ce que dit réellement la vidéo

La vidéo ne mentionne pas Engram. Elle compare AttnRes à mHC :

- mHC maintient plusieurs streams parallèles à l'intérieur de chaque couche ;
- AttnRes permet à une couche de relire sélectivement des représentations
  produites plus tôt dans la profondeur ;
- Bycloud les décrit comme deux directions théoriquement combinables, mais
  avertit que leur empilement pourrait produire des rendements décroissants.

La combinaison `Engram + AttnRes` ci-dessous est donc une hypothèse propre au
programme expérimental du dépôt, pas une conclusion de la vidéo ou du papier.

## Pourquoi tester avec Engram

Engram injecte une représentation de mémoire statique dans quelques couches.
AttnRes pourrait permettre aux couches ultérieures de réutiliser directement
les états qui contiennent cette injection, au lieu de dépendre de sa
propagation additive à travers toute la profondeur.

L'hypothèse testable est :

> AttnRes améliore-t-il la conservation et la réutilisation des informations
> injectées par Engram sans nécessiter plusieurs streams mHC ?

Risques et limites :

- les petits modèles de ce dépôt sont beaucoup moins profonds que Kimi Linear,
  donc le problème de dilution peut y être faible ;
- Full AttnRes conserve plusieurs états par token et augmente la mémoire ;
- Engram et AttnRes peuvent apprendre des fonctions redondantes ;
- la comparaison n'a de sens qu'après correction du chemin Engram fidèle au
  papier et de son décodage incrémental.

## Ablation proposée

Utiliser les mêmes plans déterministes, seeds et budgets pour :

1. baseline avec résiduel standard ;
2. AttnRes sans Engram ;
3. Engram fidèle au papier avec résiduel standard ;
4. Engram fidèle au papier avec AttnRes ;
5. optionnel : Engram fidèle au papier avec mHC comme contrôle multibranche.

Sur les modèles `tiny` peu profonds, commencer par Full AttnRes. Tester Block
AttnRes seulement sur une configuration assez profonde pour former plusieurs
blocs utiles. Rapporter perte LM, perplexité, tâches aval, mémoire maximale,
débit et latence incrémentale.
