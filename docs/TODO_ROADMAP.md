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
- [ ] Documenter versions Python/PyTorch/Transformers et commande de repro.
- [x] Ajouter un rapport automatique : loss, reward, latence, mémoire, FLOPs,
  énergie si disponible.

## M0 — Routeur

- [x] Implémenter l'interface typed du routeur et le mode déterministe `off`.
- [ ] Implémenter un routeur MLP top-k sur un pool d'experts fixe.
- [ ] Journaliser indices, poids, entropie, charge, overflow et switch rate.
- [ ] Ajouter les contrôles single-expert et random-router.

## M1 — Agent multimodal minimal

- [ ] Construire un environnement contrôlé Gymnasium avec observations image,
  symboliques et texte/octet, actions et transitions connues.
- [ ] Implémenter encodeurs séparés et projection vers un espace latent commun.
- [ ] Ajouter une policy/value et une boucle d'entraînement RL minimale.
- [ ] Créer tâches de prédiction, mémoire, intervention et planification courte.
- [ ] Établir les scores baseline et les tests de transfert inter-modalités.

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
- [ ] Ajouter gate de halting avec budget maximal et sortie anticipée.
- [ ] Mesurer gain par itération, stabilité des normes, calibration et latence.

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

## M2 — Quantification ternaire (objectif C)

- [ ] Implémenter fake-quantisation STE vers `{-1, 0, +1}` avec échelles.
- [ ] Comparer FP16/BF16, post-training quantization et QAT.
- [ ] Tester calendrier QAT progressif avec la transition ReLU² → GELU².
- [ ] Mesurer perte, perplexité/reward, dérive, activation range et mémoire.
- [ ] Ajouter un kernel ternaire de référence avant toute revendication hardware.
- [ ] Mesurer latence/énergie réelle sur CPU et GPU compatibles.

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
