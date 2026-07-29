# Multimodal Roadmap (DeepSeek OCR-like)

## Priorités architecturales majeures

Ces axes sont désormais des objectifs principaux du dépôt, distincts des
benchmarks OCR et des briques d'ablation.

### Principe d'architecture modulaire

Chaque brique doit être remplaçable et désactivable sans modifier les autres
interfaces. Le pipeline de référence sera composé de :

`observation encoder → latent streams → memory (Engram) → predictor (JEPA) →
recurrent core (Loop Transformer) → router/MoE-LoRA → policy/value → quantizer`.

Chaque étape expose une configuration explicite, des tenseurs d'entrée/sortie
documentés et un contrôle `off` servant de baseline. Les expériences doivent
permettre les combinaisons par fichier de configuration, enregistrer les
paramètres actifs et comparer chaque ajout seul avant toute combinaison.

Une brique n'est retenue que si son gain persiste dans son ablation isolée,
avec budget de données, paramètres actifs et FLOPs contrôlé. Les checkpoints,
graines et métriques doivent rester compatibles entre variantes afin de rendre
les régressions réversibles et auditables.

### A — Agent MoE-LoRA à croissance dynamique

Construire un agent RL multimodal dont la capacité augmente pendant
l'entraînement : un routeur MoE sélectionne des experts LoRA spécialisés, puis
déclenche sur nouveauté ou saturation la naissance de nouveaux experts. Chaque
expert suit un cycle de vie explicite : spécialisation, maturation, EMA des
experts stables, puis retrait ou rollback sous contrainte de budget. La
comparaison de référence sera un modèle à capacité fixe avec le même budget de
calcul cumulé.

Critères de validation : reward et regret après changement de tâche, oubli
catastrophique, interférence entre domaines, équilibre du routeur, nombre
d'expansions, paramètres actifs, coût cumulé et temps de récupération.

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

### D — Agent multimodal latent JEPA × Simulus

Construire l'agent complet autour d'un espace latent partagé pour texte,
octets, images et documents. Un encodeur transforme chaque observation en
flux latent modulaire ; un prédicteur JEPA/Simulus anticipe les états futurs
conditionnés par les actions ; une politique, une valeur ou un planificateur
utilise ces états pour agir. L'objectif est l'apprentissage d'un modèle du
monde exploitable, pas seulement l'amélioration d'un proxy de langage.

Le premier environnement sera contrôlé (observations visuelles et
symboliques, actions et transitions connues), avant un transfert vers des
documents et scènes plus réalistes. Les comparaisons incluront reconstruction
pixel, JEPA sans action, JEPA action-conditionné et flux modulaires Simulus.

Critères de validation : prédiction d'état latent, réussite de planification,
robustesse aux changements de dynamique/caméra/composition, transfert entre
modalités, mémoire latente retenue, reward et coût par étape.

Statut : **objectif central, environnement et boucle d'action à construire**.

### Matrice d'ablation interne

Dans cet environnement, toutes les architectures seront entraînées avec le
même encodeur d'entrée, la même policy/value, le même nombre de paramètres
actifs et le même budget de transitions :

1. baseline Transformer sans mémoire latente dédiée ;
2. baseline + Engram ;
3. JEPA latent action-free ;
4. JEPA latent conditionné par action ;
5. Engram + JEPA ;
6. flux modulaires Simulus/M³ ;
7. JEPA + Simulus + Loop Transformer.

Chaque ligne sera comparée sur prédiction d'état, réussite de planification,
rappel long, transfert entre modalités, reward, latence, mémoire et énergie.
Les variantes combinées ne seront conservées que si leur gain subsiste face à
leurs contrôles séparés et à un modèle de capacité équivalente.

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
