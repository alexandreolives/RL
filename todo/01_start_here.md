# Start Here

Objectif:
- définir un environnement de test suffisant
- puis définir un premier système crédible qui puisse viser un très bon rapport perf / compute sous `<= 40GB`

## Todo immédiate

- [ ] Définir l’environnement de test de départ.
  - assez dur pour discriminer les bonnes idées des gadgets
  - assez rapide pour itérer sous budget
  - assez standard pour permettre une comparaison honnête

- [ ] Fixer l’environnement matériel cible exact.
  - GPU exact
  - VRAM exacte
  - CPU / RAM
  - temps d’entraînement acceptable

- [ ] Choisir le benchmark de départ.
  - `Atari / ALE`
  - `DMControl`
  - `Procgen`
  - `Crafter`
  - `gridworld / puzzles`
  - autre

- [ ] Fixer la vraie métrique d’optimisation.
  - score brut
  - score par heure GPU
  - score par nombre d’interactions environnement
  - score par GB de VRAM

- [ ] Choisir la baseline de départ.
  - `Dreamer-style`
  - `M^3-style`
  - `JEPA/world model compact`

- [ ] Décider si on fait:
  - from scratch
  - adaptation d’un repo existant
  - reproduction minimale puis modifications

## Recommandation actuelle

Premier choix raisonnable:
- benchmark principal: `Crafter`
- benchmark de contrôle: `DMControl`
- partir d’une baseline `Dreamer-style`
- la rendre plus compacte et plus stable
- garder `SwiGLU` comme activation candidate principale
- comparer ensuite contre une variante `GELU`
- autoriser une variante `MoE` simple dès la v1

Pourquoi:
- `Crafter` est plus aligné avec l’ambition long-horizon / monde structuré
- `DMControl` sert de sanity check rapide et propre
- c’est le chemin le plus défendable techniquement
- Dreamer est un point d’ancrage RL/world model clair
- on garde une architecture lisible même avec un `MoE` simple

## Livrable attendu

À la fin de cette étape, on doit avoir un document très court qui répond à:

- quel environnement de test on choisit
- quelle tâche on vise
- quelle baseline on prend
- quel budget exact on s’autorise
- quelle métrique on optimise
