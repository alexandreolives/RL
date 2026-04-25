# Experiment Protocol

Le but de cette todo est d’éviter les expériences floues.

## Avant toute ligne de code

- [ ] écrire le tableau des hyperparamètres de base
- [ ] écrire le budget max par run
- [ ] écrire le critère de succès minimal
- [ ] écrire le baseline exact à battre

## Première série d’expériences

- [ ] Run 0: benchmarker l’environnement de test lui-même
  - vitesse
  - coût
  - variance
  - longueur d’épisode

- [ ] Run A: baseline dense `Dreamer-style` + `GELU` sur `Crafter`
- [ ] Run B: baseline dense `Dreamer-style` + `SwiGLU` sur `Crafter`
- [ ] Run C: variante `MoE simple` + `SwiGLU` sur `Crafter`
- [ ] Run D: meilleure variante rejouée sur `DMControl`

## Mesures à logguer obligatoirement

- [ ] VRAM max
- [ ] steps/sec
- [ ] temps par epoch / par tranche d’interaction
- [ ] score environnement
- [ ] loss world model
- [ ] qualité des reconstructions / prédictions si applicable
- [ ] stabilité de l’entraînement

## Questions auxquelles chaque run doit répondre

- [ ] est-ce que ça rentre vraiment dans `<= 40GB` ?
- [ ] est-ce que le gain vient de la meilleure architecture ou juste d’un run plus long ?
- [ ] est-ce que la perf est due à plus de compute online ou à un meilleur world model ?
- [ ] est-ce que l’activation change vraiment quelque chose ?
- [ ] est-ce que le MoE aide vraiment à budget actif comparable ?

## Règle de discipline

Ne modifier qu’un petit nombre de variables à la fois.

Priorité:

1. baseline stable
2. comparaison activation
3. comparaison taille / coût
4. seulement après: modifications plus ambitieuses
