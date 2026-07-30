# Réévaluation hybride et résiduelle

## Implémentation

`ConfigurableHybridCore` permet maintenant de définir, étage par étage, une
séquence de `fourier`, `kda`, `attention`, `loop` ou `mhc`. Chaque étage accepte
`repeats`, et `iterations` réapplique toute la séquence. `return_trace=True`
archive le chemin réellement exécuté. Le module est exposé dans
`ModularMultimodalAgent` par `hybrid_stages` et `hybrid_iterations`.

Exemple :

```python
hybrid_stages=["fourier", HybridStage("kda", repeats=2), "loop", "attention"]
agent = ModularMultimodalAgent(hybrid_stages=hybrid_stages, hybrid_iterations=2)
```

## Vérification courante

- Suite complète : **61 tests passent**.
- Tests résiduels et DeepSeek ciblés : **16 tests passent**.
- Le benchmark `scripts/benchmark_k3_blocks.py` couvre KDA, QAT, hybride
  Kimi, LatentMoE et les schedules Fourier/KDA/attention et
  Fourier/KDA/Loop/attention.
- `scripts/benchmark_hybrid_schedules.py` compare attention-only,
  Fourier→attention, Fourier→KDA→attention, Fourier→KDA→Loop→attention et
  KDA 3:1 pour plusieurs nombres d'itérations.
- `AdaptiveHybridCore` et les paramètres `hybrid_fast_stages` /
  `hybrid_full_stages` permettent un choix dynamique piloté par l'incertitude ;
  le chemin sélectionné est inclus dans la trace.

Ces résultats vérifient la construction, les gradients et les interfaces ; ils
ne constituent pas encore une comparaison qualité scientifique. Les prochains
runs doivent apparier seeds, paramètres actifs, FLOPs et longueur de contexte
pour baseline, Engram, mHC, Full Attention Residual, DeepSeek et les schedules
hybrides.

## Contrôle adversarial du score nul

Le générateur d'évaluation tire de nouvelles séquences ; `--fixed-batch` ne
contrôle que l'entraînement. Sur un overfit contrôlé (baseline, tiny,
symbolique, longueur 40, batch 4, 100 étapes, CPU), le modèle atteint
`train_acc_mean_last10=1.0` tandis que l'évaluation aléatoire reste `0.0`.
Le même comportement est observé en GPU après 20 étapes. Le `0` des runs de
10 étapes ne signalait donc pas un oubli de gradient : il combinait un budget
trop court et une évaluation hors-échantillon. Toute future comparaison doit
rapporter séparément train/fixed-batch, validation fixe et validation aléatoire.

## Smoke benchmark CPU (31 juillet 2026)

Séquence 8, batch 1, eager, un warmup et une mesure :

| variante | paramètres | ms/run | tokens/s |
|---|---:|---:|---:|
| baseline | 40.09M | 14.06 | 568.94 |
| AttnRes | 40.11M | 27.76 | 288.21 |
| Engram no-conv | 51.14M | 26.45 | 302.41 |
| Engram no-conv + AttnRes | 51.16M | 34.50 | 231.91 |
| mHC | 48.58M | 25.31 | 316.15 |
| DeepSeek v6 + Engram | 273.26M | 407.61 | 19.63 |
| DeepSeek v6 + Engram + AttnRes | 273.28M | 740.24 | 10.81 |

Ce smoke test montre le coût supplémentaire de l'agrégation résiduelle, surtout
sur le chemin v6 ; il ne permet pas de conclure sur la qualité. Les mesures
longues et multi-seeds restent obligatoires.

Un run court reproductible (2 seeds, tiny, symbolique, longueur 40, 5 étapes)
a aussi été relancé avec `baseline`, `attnres`, `engram_noconv`,
`engram_noconv_attnres` et `mhc`. Les accuracies passkey moyennes obtenues sont
respectivement `0.125`, `0.125`, `0.0`, `0.0` et `0.0` ; ce budget est trop court
pour conclure sur la qualité, mais confirme que les chemins s'entraînent et
produisent des métriques comparables.

## Smoke benchmark GPU WSL2

Copie isolée du dépôt (`/tmp/rl-m1-current`) dans l'image `rl-m1:fast`,
PyTorch `2.6.0+cu124`, RTX 3060 Ti, séquence 32, batch 2, deux mesures :

| schedule | 1 itération (ms) | 2 itérations (ms) |
|---|---:|---:|
| attention-only | 0.74 | 1.29 |
| Fourier→attention | 0.73 | 1.10 |
| Fourier→KDA→attention | 14.65 | 28.67 |
| Fourier→KDA→Loop→attention | 20.80 | 40.77 |
| KDA 3:1→attention | 80.54 | 96.88 |

Le conteneur GPU ne contient pas `pytest`; la validation fonctionnelle reste
faite localement (**64 tests**). Le benchmark CUDA confirme que la copie
isolée contient le code courant et que les schedules s'exécutent sur le GPU.

La campagne d'entraînement multi-seed n'a pas pu démarrer dans cette image :
son `transformers` ne fournit pas `DeepseekV4Config`, importé par le chemin
DeepSeek du dépôt. Elle reste exécutable dans l'environnement local ; il faut
reconstruire/mettre à jour l'image avant de considérer une mesure GPU qualité
comme valide.

Après reconstruction avec `transformers==5.13.1`, la campagne CUDA a démarré
correctement (RTX 3060 Ti, tiny, longueur 64, 2 seeds, 10 étapes). Les
accuracies passkey sont restées à `0.0` pour baseline, AttnRes, Engram
no-conv et Engram no-conv + AttnRes ; le budget est court et ne permet pas de
conclure sur la qualité, mais valide le pipeline GPU de bout en bout.

## Grille KDA × Loop sur GPU

La grille complète (13 schedules, 1/2 itérations, état KDA conservé) a été
exécutée sur RTX 3060 Ti, séquence 32. En latence seule, les candidats les
plus rapides sont Fourier→attention (`0,62 ms`, 1 itération), attention-only
(`0,63 ms`, 2 itérations) et Loop-only (`1,13 ms`, 1 itération). Les chemins
KDA coûtent nettement plus cher dans cette référence Python : KDA→Loop
(`16,11 ms`) et KDA→Loop→attention (`20,12 ms`).

Ce classement est uniquement un filtre de coût : il ne désigne pas le meilleur
modèle. Les candidats retenus pour la phase qualité sont Fourier→attention,
Loop-only, KDA→Loop et KDA→Loop→attention, chacun avec état KDA réinitialisé
et conservé, puis avec 1–4 itérations.

## Qualité multimodale (2 seeds, 30 épisodes)

Sur `MultimodalMemoryEnv`, avec le même budget et les mêmes seeds :

| variante | visible moyen | hidden moyen |
|---|---:|---:|
| baseline | 0,625 | 0,600 |
| Loop-only | 0,350 | 0,600 |
| Fourier→attention | **0,650** | 0,400 |
| KDA→Loop | 0,350 | 0,600 |
| KDA→Loop→attention | 0,350 | 0,600 |

Résultat provisoire : Fourier→attention gagne légèrement sur le signal visible,
mais perd sur le rappel caché ; la baseline reste meilleure globalement. Les
variantes KDA×Loop ne montrent aucun gain avec seulement 30 épisodes. Elles ne
doivent donc pas être retenues comme “optimales” avant un entraînement plus long
et des tâches de rappel long dédiées.

Un second run (100 épisodes, seeds 0/1, même environnement) donne les moyennes
suivantes : baseline `visible=0,40 / hidden=0,525`, Loop-only
`0,40 / 0,525`, Fourier→attention `0,575 / 0,425`, KDA→Loop
`0,60 / 0,475`, KDA→Loop→attention `0,40 / 0,525`. Le score composite
visible+hidden favorise provisoirement KDA→Loop (`1,075` contre `0,925` pour
la baseline), mais l'écart repose sur deux seeds et doit être confirmé.

Le troisième seed (baseline, Fourier→attention, KDA→Loop, 100 épisodes) donne
respectivement `(visible, hidden)=(0,700,0,500)`, `(0,400,0,500)` et
`(0,600,0,500)`. Sur les **3 seeds**, les moyennes deviennent : baseline
`0,500 / 0,517` (composite `1,017`), Fourier→attention `0,517 / 0,450`
(`0,967`) et KDA→Loop `0,600 / 0,483` (`1,083`). KDA→Loop est donc le meilleur
score composite actuel, mais son avantage reste exploratoire et doit être
confirmé sur des tâches de rappel long et davantage de seeds.

Enfin, le screening exhaustif qualité (20 épisodes, seeds 0/1/2) donne les
meilleurs composites ex aequo à `1,133` pour KDA→Fourier→Loop et KDA→Loop avec
état conservé ; Fourier→attention atteint `1,117`, la baseline `1,100`.
Ces scores sont plus bruités que la campagne 100 épisodes et ne suffisent pas
à choisir entre les deux architectures gagnantes. La phase suivante doit
entraîner ces deux candidates, plus la baseline, sur 100–500 épisodes et des
longueurs de séquence croissantes.

Un screening de stabilité (`scripts/benchmark_hybrid_stability.py`) sur toute
la grille, 1–4 itérations et état conservé, a trouvé **0 sortie non finie**.
Les gradients les plus élevés apparaissent sur KDA→Loop à 4 itérations
(`max_param_grad≈12,95`) : cette configuration doit être surveillée pendant
l'entraînement, même si elle reste numériquement finie.
