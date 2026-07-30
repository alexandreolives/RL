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
