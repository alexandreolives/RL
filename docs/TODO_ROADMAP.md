# TODO technique — Agent multimodal adaptatif

Ce document transforme la vision du projet en jalons vérifiables. Un jalon ne
doit être marqué terminé que si le code, la configuration, les tests et les
résultats sont archivés.

## M0 — Socle commun et auditabilité

- [x] Définir les contrats tensoriels entre encodeur, flux latent, mémoire,
  prédicteur, boucle, routeur et policy/value.
- [x] Ajouter des configs versionnées pour chaque composant et un mode `off`.
- [x] Fixer seeds, budgets de paramètres/FLOPs, protocole d'évaluation et
  format JSON des métriques.
- [x] Ajouter smoke tests CPU, test de déterminisme et vérification des shapes.
- [x] Documenter versions Python/PyTorch/Transformers et commande de repro.
- [x] Ajouter un rapport automatique : loss, reward, latence, mémoire, FLOPs,
  énergie si disponible.

## M0 — Routeur

- [x] Implémenter l'interface typed du routeur et le mode déterministe `off`.
- [x] Implémenter un routeur MLP top-k sur un pool d'experts fixe.
- [x] Journaliser indices, poids, entropie, charge, overflow et switch rate.
- [x] Ajouter les contrôles single-expert et random-router.

## M1 — Agent multimodal minimal

- [x] Construire un environnement contrôlé Gymnasium avec observations image,
  symboliques et texte/octet, actions et transitions connues.
- [x] Implémenter encodeurs séparés et projection vers un espace latent commun.
- [x] Ajouter une policy/value et une boucle d'entraînement RL minimale.
- [x] Créer tâches de prédiction, mémoire, intervention et planification courte.
- [x] Établir les scores baseline et les tests de transfert inter-modalités.

## M1 — Mémoire Engram

- [ ] Brancher Engram comme bloc optionnel après les flux latents.
- [ ] Vérifier que son cache est isolé par épisode et correctement sérialisé.
- [ ] Comparer baseline, Engram et Engram avec routeur au même budget actif.
- [ ] Mesurer rappel long, collisions, coût mémoire et latence.

## M1 — JEPA latent et world model

- [ ] Implémenter encodeur cible/online et prédiction du prochain latent.
- [ ] Ajouter les régularisateurs anti-collapse (variance/covariance/SIGReg).
- [ ] Comparer reconstruction pixel, JEPA action-free et JEPA action-conditionné.
- [ ] Mesurer erreur multi-horizon, suffisance latente et qualité de planification.

## M1 — Loop Transformer

- [ ] Implémenter un bloc Transformer partagé réappliqué `N` fois.
- [ ] Ajouter état récurrent, résidu stable, normalisation et gradient checkpoint.
- [ ] Comparer profondeur fixe, profondeur variable et modèle non récurrent.
- [x] Ajouter gate de halting avec budget maximal et sortie anticipée.
- [ ] Mesurer gain par itération, stabilité des normes, calibration et latence.

## M1 — Boucles hybrides spectral/attention

- [ ] Implémenter une architecture hybride à planning explicite, où chaque
  position de profondeur peut choisir indépendamment `Fourier/FFT`, KDA/SSM,
  attention dense, Loop Transformer ou un bloc DeepSeek/mHC.
- [ ] Permettre de modifier le type de bloc et son ratio pour chaque étage
  (`3:1`, `1:1`, `4:1`, attention finale seulement), sans changer les
  interfaces de tenseurs ni le routeur.
- [ ] Tester les schedules `spectral → attention`, `attention → spectral`,
  Fourier → KDA → attention et alternance différente à chaque itération.
- [ ] Ajouter un switch dynamique piloté par profondeur, modalité, longueur de
  contexte ou incertitude du routeur, avec journalisation du chemin réellement
  exécuté.
- [ ] Ajouter un mode Loop qui réapplique le planning complet plusieurs fois,
  avec budget maximal, halting et état récurrent contrôlable.
- [ ] Comparer à budget actif égal : attention seule, spectral seul, hybride
  fixe et hybride dynamique.
- [ ] Mesurer qualité, dépendances longues, aliasing, stabilité numérique,
  latence FFT, mémoire et coût sur séquences courtes/longues.
- [ ] Vérifier que le switch ne dégrade pas les gradients ni la calibration du
  halting ; conserver un fallback attention-only.

## M2 — Réévaluation résiduelle/DeepSeek

- [ ] Rebaser les comparaisons historiques sur l'implémentation actuelle de
  `FullAttentionResidual`, mHC et hyper-connections DeepSeek, sans réutiliser
  les anciens chiffres obtenus avec les primitives précédentes.
- [ ] Rejouer baseline, résidu séquentiel, Full Attention Residual, mHC et
  agrégation uniforme à profondeur, seeds, paramètres actifs et FLOPs égaux.
- [ ] Rejouer les combinaisons DeepSeek + Engram, DeepSeek + KDA/FFT, mHC +
  Engram et KDA + Attention Residuals, avec et sans Loop.
- [ ] Mesurer qualité, rappel long, pertes d'information, normes/gradients,
  latence, mémoire et coût de calcul ; archiver chaque configuration et seed.
- [ ] Vérifier spécifiquement si le changement de résidu modifie les résultats
  précédents, puis documenter les gains, régressions et résultats négatifs.

## M2 — MoE-LoRA à croissance dynamique (objectif A)

- [ ] Implémenter experts LoRA indépendants, capacité maximale et top-k dispatch.
- [ ] Définir un signal de nouveauté/saturation et un seuil de naissance.
- [ ] Ajouter initialisation prudente, warm-up et bonus d'exploration temporaire.
- [ ] Implémenter maturation, morphing d'activation et EMA des experts stables.
- [ ] Ajouter retraite, réactivation, rollback et budget de paramètres.
- [ ] Comparer croissance dynamique à capacité fixe à budget cumulé égal.
- [ ] Mesurer reward, regret, oubli, Gini du routage et temps de récupération.

## M2 — Routeur hiérarchique

- [ ] Séparer gates modalité, domaine, expert, profondeur et nouveauté.
- [ ] Ajouter balance loss, pénalité overflow et régularisation temporelle.
- [ ] Calibrer incertitude et prédiction du nombre de boucles.
- [ ] Comparer MLP, hiérarchique, récurrent et uncertainty-aware.
- [ ] Tester le routeur sur changements de tâche et expansion d'experts.

## M2 — Hybridation mémoire récurrente × Loop (KDA/Mamba/FFT)

- [ ] Implémenter un bloc Delta/linear-attention à état récurrent borné, inspiré
  de KDA, puis l'hybrider avec notre Loop, Engram et JEPA ; ne pas viser une
  reproduction isolée du modèle Kimi.
- [ ] Ajouter un contrôle Mamba-2/Selective State Space si une implémentation
  auditée est disponible ; ne pas appeler KDA « Mamba-2 » sans équivalence
  démontrée.
- [ ] Comparer attention quadratique, KDA/Delta, Mamba-2/SSM et FFT, puis leurs
  combinaisons avec le routeur, Engram et JEPA sur les mêmes budgets actifs.
- [ ] Tester des ratios hybrides `3:1`, `1:1` et `attention-only` avec une
  attention globale finale pour le rappel exact.
- [ ] Mesurer qualité long contexte, coût d'état récurrent, KV-cache, débit,
  stabilité et transfert multimodal.
- [ ] Brancher ces blocs dans le Loop Transformer sans modifier l'interface du
  routeur, afin de les ablater indépendamment.

## M2 — Stable LatentMoE et Attention Residuals

- [x] Ajouter un MoE sur dimension latente avec experts routés, compression et
  experts partagés.
- [ ] Tester équilibrage par quantiles des scores du routeur contre auxiliaire
  classique, avec nombre d'experts actifs contrôlé.
- [x] Ajouter AttnRes sur les représentations précédentes ; [ ] comparer avec
  résidu séquentiel,
  mHC et agrégation uniforme à profondeur égale.
- [ ] Mesurer oubli, interférence entre modalités, stabilité du gradient et
  mémoire de KV/activations.

Inspirations Kimi K3 / MoonEP à hybrider avec notre stack (source archive:
`youtube/bycloud/g683I1-4MKE_A Closer Look At Kimi K3s INSANE Architecture Breakthrough/`):

- [x] Implémenter une référence KDA minimale (delta-rule, état borné,
  décroissance par canal).
- [ ] Tester KDA+Loop, KDA+Engram, KDA+JEPA et KDA+spectral ; mesurer
  collisions et rappel long.
- [ ] Comparer les hybrides KDA/full-attention `3:1`, `1:1`, KDA+Loop et
  attention-only à FLOPs actifs, qualité, KV-cache et latence appariés.
- [x] Ajouter un benchmark reproductible paramètres/latence des variantes
  KDA, hybride, QAT et LatentMoE ; [ ] compléter mesures GPU/FLOPs.
- [ ] Ajouter un fallback global périodique/final piloté par l'incertitude pour
  récupérer les associations exactes perdues par l'état compressé.
- [x] Implémenter une référence LatentMoE avec compression avant dispatch,
  expansion après expert et top-k ; [ ] tester ses variantes puis l'hybrider
  avec nos experts LoRA, Engram et le routeur top-k ; mesurer
  balance par quantiles, qualité et octets de communication.
- [x] Ajouter une réplication dynamique locale des experts MoonEP indépendamment
  du routeur appris ; [ ] mesurer overflow, charge par rang et coût de
  synchronisation sur plusieurs GPU.
- [ ] Tester QAT MXFP4/MXFP8 sur les chemins Loop/KDA/FFT et les experts LoRA,
  avec curriculum long contexte `8K→64K→256K→1M`, après contrôle BF16.
- [ ] Documenter séparément post-training multi-teacher, effort de raisonnement
  variable et entraînement vision-langage natif ; ne pas les mélanger aux
  ablations d'architecture.

## M2 — Quantification ternaire (objectif C)

- [x] Implémenter une fake-quantification bloc MXFP4/MXFP8 avec STE et échelles.
- [ ] Comparer FP16/BF16, post-training quantization et QAT.
- [ ] Tester calendrier QAT progressif avec la transition ReLU² → GELU².
- [ ] Mesurer perte, perplexité/reward, dérive, activation range et mémoire.
- [ ] Ajouter un kernel ternaire de référence avant toute revendication hardware.
- [ ] Mesurer latence/énergie réelle sur CPU et GPU compatibles.

## M2 — Calcul adaptatif spectral + QAT

- [ ] Mesurer séparément le gain du chemin FFT (coût moyen) et la perte due à
  la quantification (qualité par boucle/token).
- [ ] Appliquer la QAT au chemin spectral et au chemin attention, puis tester
  un switch FFT→attention avec poids/activations quantifiés.
- [ ] Entraîner la gate sur un budget de calcul ou d'énergie, pas seulement sur
  la loss, avec pénalité si l'attention est activée inutilement.
- [ ] Comparer qualité moyenne, pire cas, taux d'activation attention, latence,
  mémoire et énergie à budget fixe.
- [ ] Garder un chemin haute précision attention-only comme contrôle de qualité.
- [ ] Tester séparément la compatibilité avec un backbone de diffusion :
  débruitage spectral initial, attention conditionnelle tardive et nombre
  adaptatif d'étapes. Ne pas extrapoler les résultats agent/LLM à la diffusion.

## M2 — Prédiction multi-token / Markov et experts diffusion

- [ ] Ajouter des têtes MTP `k=2,4,8` sur le tronc partagé, avec perte auxiliaire
  et contrôle next-token-only.
- [ ] Ajouter les configurations de décodage par blocs `B=8` et `B=16`, avec
  partage des états/KV et comparaison à `B=1`.
- [ ] Tester une factorisation causale des tokens futurs, une tête indépendante
  et une dépendance Markov/probabiliste entre propositions.
- [ ] Ajouter vérification spéculative par le modèle principal et mesurer taux
  d'acceptation, tokens acceptés par passe, qualité et accélération réelle.
- [ ] Ajouter un score de confiance par position et un arrêt précoce lorsque
  la probabilité d'acceptation ou la confiance Markov conditionnelle tombe sous
  un seuil ; comparer seuil fixe, seuil calibré et budget adaptatif.
- [ ] Tester l'arrêt au token `j` dans chaque bloc (`1 ≤ j ≤ B`) et le fallback
  immédiat au bloc suivant lorsque `j` est rejeté.
- [ ] Mesurer longueur acceptée, distribution des arrêts, taux de rejet,
  qualité de l'échantillonnage et coût de la vérification, sans confondre
  confiance du draft avec probabilité correcte du modèle cible.
- [ ] Adapter le mécanisme au byte-level et aux flux latents multimodaux.
- [ ] Implémenter un expert diffusion latent à 1–4 étapes avec conditionnement
  timestep, puis le router seulement sur les états incertains.
- [ ] Comparer expert diffusion isolé, expert Transformer et combinaison
  diffusion→attention dans le Loop Transformer.
- [ ] Mesurer coût moyen/pire cas, calibration, cohérence temporelle et qualité
  sans attribuer à tort les résultats à Z.ai ou à un papier particulier.

## M2 — DSpark : vérification semi-autoregressive adaptative

- [ ] Implémenter un drafter parallèle qui propose un bloc de `B=8` puis `B=16`
  tokens depuis un tronc partagé.
- [ ] Ajouter un petit module séquentiel intra-bloc pour réintroduire la
  dépendance entre tokens proposés et mesurer la suffix decay.
- [ ] Estimer la probabilité de survie de chaque préfixe et vérifier seulement
  le préfixe utile ; comparer à la vérification du bloc complet.
- [ ] Calibrer le seuil selon difficulté, longueur de contexte et charge/
  throughput du serveur, sans changer la distribution finale vérifiée.
- [ ] Comparer MTP indépendant, DSpark-like, EAGLE et autoregressif à qualité
  identique : longueur acceptée, tokens/passe, latence p50/p95, débit et coût.
- [ ] Réutiliser les KV/index du premier pas lorsque possible et mesurer le
  gain mémoire séparément du gain de calcul.
- [ ] Tester l'analogie avec le Loop Transformer : FFT/SSM/routeur rapide en
  proposition, attention récurrente en vérification, puis halting adaptatif.

## M2 — Multimodal documents/OCR

- [ ] Finaliser ingestion PDF/image, OCR, layout et ordre de lecture.
- [ ] Produire un schéma JSON canonique et dataset smoke reproductible.
- [ ] Brancher Engram, JEPA et flux latents sur documents.
- [ ] Évaluer extraction factuelle, raisonnement, robustesse visuelle et VRAM.
- [ ] Tester compression en hyper-tokens et récupération factuelle.

## M3 — Intégration JEPA × Simulus (objectif D)

- [ ] Implémenter flux latents modulaires état/mémoire/contrôle.
- [ ] Conditionner le prédicteur futur sur l'action et l'état récurrent.
- [ ] Ajouter rollout latent, planificateur et policy model-based.
- [ ] Comparer architecture monolithique et streams modulaires M³/Simulus.
- [ ] Tester horizons longs, contre-factuels et changements de dynamique/caméra.
- [ ] Intégrer ensuite Engram, Loop Transformer, routeur dynamique et policy.

## M3 — Validation scientifique et publication

- [ ] Refaire chaque résultat avec au moins 5 seeds et intervalles de confiance.
- [ ] Tester contextes 256/512/2048, byte-level et tokenisé.
- [ ] Utiliser plusieurs tailles de modèle et budgets FLOPs/paramètres appariés.
- [ ] Publier courbes d'apprentissage, variance, énergie et échecs négatifs.
- [ ] Geler configs/checkpoints nécessaires et générer un rapport automatique.
- [ ] Vérifier dépôt public : secrets, licences, données et reproductibilité.
- [ ] Préparer un rapport technique ou preprint sans extrapoler les proxys.

## Critère de passage entre jalons

Un composant passe de M1 à M2 seulement si son ablation isolée est stable,
reproductible et utile à budget contrôlé. Une combinaison passe à M3 seulement
si son gain reste présent face à chaque bloc seul et à un contrôle de capacité
équivalente.
