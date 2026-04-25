# Plan Compression d'Information (JEPA + OCR + Hyper-Tokens)

Ce document formalise le plan discuté:
- JEPA pour la compréhension latente
- OCR visuel pour fidélité des données
- compression sémantique en hyper-tokens
- architecture hybride à deux voies

## 1) Objectif

Construire un système multimodal qui:
- compresse fortement l'information utile
- garde la précision des éléments critiques (noms, montants, dates)
- reste exécutable sur budget compute/mémoire réduit

## 2) Architecture cible

Deux voies parallèles:

- voie A (cognitive):
  - encodeur visuel/texte -> espace latent JEPA
  - prédiction latente (pas reconstruction pixel/token)
  - sortie: hyper-tokens sémantiques compacts

- voie B (factualité brute):
  - OCR/document parser haute fidélité
  - extraction explicite champs sensibles
  - sortie: mémoire factuelle structurée (tables/kv/spans)

Fusion:
- module de routage qui choisit A, B, ou A+B selon la requête.

## 3) Hyper-Tokens (compression)

Principe:
- remplacer une longue séquence token-level par un petit ensemble de vecteurs
  représentant concepts, relations et contraintes.

Contraintes:
- conserver traçabilité vers la source (offset/page/zone)
- limiter la perte sur infos exactes
- permettre une expansion contrôlée (dézippage) vers une réponse lisible

## 4) Entraînement proposé

Phase 1 (pré-entraînement représentation):
- JEPA sur vues masquées/multi-vues
- perte de prédiction latente
- régularisation anti-collapse (EMA ou LeJEPA-style isotropic constraints)

Phase 2 (alignement compressif):
- objectif de compaction:
  - minimiser taille latente sous contrainte de performance downstream
- distillation:
  - teacher long-context -> student compact hyper-tokens

Phase 3 (hybridation avec factualité):
- apprentissage du routeur A/B
- pertes mixtes:
  - sémantique (qa/résumé/raisonnement)
  - exactitude factuelle (entity/value exact match)

## 5) Mécanisme de "dézippage sémantique"

Pour répondre en langage naturel:
- decodeur conditionné par hyper-tokens + mémoire factuelle
- garde-fous:
  - priorité aux champs factuels de la voie B
  - citation de provenance (span/page) pour assertions sensibles

## 6) Métriques

Compression:
- ratio de compression latent/token
- coût mémoire VRAM/RAM
- latence inférence

Qualité:
- exact match sur champs factuels
- benchmarks de raisonnement/qa
- robustesse long-context

Stabilité:
- score anti-collapse (variance latente, isotropie, uniformité)
- drift entre seeds/runs

## 7) Risques techniques

- collapse des représentations latentes
- perte d'information lors de compaction agressive
- conflits entre objectifs "abstraction" et "exactitude brute"
- surcoût de fusion si routage mal calibré

## 8) Roadmap pratique (itérative)

1. baseline OCR + LLM standard, métriques de référence
2. ajout encodeur JEPA latent-only
3. prototype hyper-token bottleneck
4. routeur A/B + éval fidélité
5. optimisation pour infra contrainte (edge/VRAM limitée)

## 9) Sortie attendue

Un système:
- plus compact qu'un pipeline token-level pur
- plus précis qu'un JEPA seul sur données factuelles
- plus robuste en long contexte grâce à la compression structurée.
