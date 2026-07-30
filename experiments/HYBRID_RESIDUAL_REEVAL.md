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
