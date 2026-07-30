# Kimi K3 — implémentation expérimentale

Cette branche ajoute des briques **inspirées des publications Kimi**, sans
prétendre reproduire leur kernel de production :

- `models.atoms.KDA` : attention linéaire récurrente delta-rule avec état
  streaming et décroissance par canal ;
- `HybridKDA` : point de comparaison KDA + attention dense ;
- `LatentMoE` : compression avant dispatch, top-k experts dans l'espace latent,
  puis expansion ; experts partagés optionnels (`shared_experts`) et
  `last_aux_loss` exposant un signal de balance ; `rebalance_replicas()` ajoute
  des slots redondants pour les experts surchargés (analogue local à MoonEP) ;
- `use_attn_res=True` : agrégation Attention Residuals des sorties Engram,
  spectral, LatentMoE, KDA et Loop dans l'agent ;
- `MXFP4FakeQuant` / `MXFP8FakeQuant` : fake-quantification bloc avec STE
  pour préparer les ablations QAT ; `KDA(qat=True)` active la quantification
  des activations MXFP8.
- `ModularMultimodalAgent(use_kda=True, use_latent_moe=True)` permet de tester
  ces blocs indépendamment, sans modifier la configuration historique.

## Limites et protocole

La KDA fournie est une référence PyTorch lisible (boucle temporelle), pas une
implémentation optimisée Triton/FlashAttention. Les performances doivent donc
être mesurées séparément de la correction fonctionnelle. Les prochains tests
doivent comparer : attention dense, KDA seule, ratio 3:1 KDA/dense, puis
LatentMoE avec mêmes paramètres, longueur de contexte, seed et budget FLOPs.

Références primaires : Kimi Linear (`arXiv:2510.26692`), rapport Kimi K3
(`arXiv:2607.24653`), LatentMoE (`arXiv:2601.18089`).

La fake-quantification ne prétend pas encoder les octets MXFP ni fournir le
kernel accéléré ; elle sert à comparer la stabilité et la qualité en entraînement.
