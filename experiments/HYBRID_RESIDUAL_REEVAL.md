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

Ces résultats vérifient la construction, les gradients et les interfaces ; ils
ne constituent pas encore une comparaison qualité scientifique. Les prochains
runs doivent apparier seeds, paramètres actifs, FLOPs et longueur de contexte
pour baseline, Engram, mHC, Full Attention Residual, DeepSeek et les schedules
hybrides.
