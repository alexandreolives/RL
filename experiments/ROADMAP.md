# Multimodal Roadmap (DeepSeek OCR-like)

## Priorités architecturales majeures

Ces deux axes sont désormais des objectifs principaux du dépôt, distincts des
benchmarks OCR et des briques d'ablation.

### A — Modèle à croissance dynamique

Construire un agent RL dont la capacité augmente pendant l'entraînement :
détection de nouveauté ou de saturation, naissance d'experts, spécialisation,
maturation, EMA des experts stables, puis retrait ou rollback sous contrainte
de budget. La comparaison de référence sera un modèle à capacité fixe avec le
même budget de calcul cumulé.

Critères de validation : reward et regret après changement de tâche, oubli
catastrophique, équilibre du routeur, nombre d'expansions, paramètres actifs,
coût cumulé et temps de récupération.

Statut : **planifié, non implémenté**.

### B — Loop Transformer / réseau récurrent

Implémenter un bloc Transformer partagé, réappliqué plusieurs fois au même
état latent, avec profondeur fixe puis profondeur adaptative via une gate de
halting. L'objectif est de mesurer si la récurrence permet une amélioration
compute-adaptive du raisonnement et de la planification, sans attribuer à la
boucle une capacité implicite non démontrée.

Critères de validation : comparaison non récurrente à FLOPs actifs égaux,
performance selon le nombre d'itérations, calibration de l'arrêt, stabilité
des normes latentes, coût/latence et robustesse sur dépendances longues.

Statut : **planifié, module générique à implémenter**.

### C — Quantification ternaire / 1,58 bit

Construire une voie QAT progressive vers des poids ternaires
`{-1, 0, +1}`, avec facteurs d'échelle appris. Le modèle haute précision
servira de contrôle ; la contrainte sera introduite graduellement avant une
éventuelle étape de déploiement avec kernel ternaire dédié. Cette voie doit
être testée séparément sur le Transformer dense, le Loop Transformer et les
experts dynamiques afin d'identifier les interactions réelles.

Critères de validation : perte et perplexité, dérive numérique, distribution
des poids/activations, paramètres et mémoire, latence, énergie par token ou
par étape RL, et rapport qualité/coût face à FP16/BF16 et à la quantification
post-entraînement. Aucun gain matériel ne sera revendiqué sans kernel ou
mesure hardware correspondante.

Statut : **planifié, kernel ternaire et QAT à implémenter**.

## Status global

- `Step 1`: done (baseline/engram/engram_noconv text-byte)
- `Step 2`: done; the paper-aligned nine-seed text-byte evaluation found a
  small, statistically inconclusive regression against `engram_noconv`, which
  remains the preferred default
- `Step 3`: in progress (OCR-like foundation)

## Step 3 — OCR-like Foundation (`experiments/step3_ocr_like`)

Goal:
- build a minimal document pipeline inspired by DeepSeek OCR:
  - ingest PDF/image
  - extract text + layout + reading order
  - produce normalized doc records

Outputs:
- scripts for extraction
- canonical JSON schema
- smoke dataset + sanity checks

## Step 4 — Engram Integration (`experiments/step4_engram_integration`)

Goal:
- plug `engram_noconv` into the doc pipeline representation path.

Outputs:
- training/eval scripts for doc representation
- configs for deterministic runs

## Step 5 — Hyper-token Compression (`experiments/step5_hypertoken_compression`)

Goal:
- compress doc semantics into dense hyper-tokens while keeping factual recovery.

Outputs:
- compression module
- reconstruction/factual probes

## Step 6 — Doc Benchmark (`experiments/step6_doc_benchmark`)

Goal:
- compare:
  - baseline doc pipeline
  - `engram_noconv`
  - `engram_noconv + perceptual lejepa`

Metrics:
- factual EM/F1
- reasoning accuracy
- latency + VRAM
