# Architecture Choices

Questions à trancher avant d’implémenter.

## World model

- [ ] Latent continu ou tokens discrets ?
- [ ] RSSM / Dreamer classique ou design plus modulaire type `M^3` ?
- [ ] Modèle très compact et stable type `LeWorldModel` ou modèle plus standard à imaginer ?

## Policy / RL

- [ ] Actor-critic dans l’imagination
- [ ] planning explicite
- [ ] hybridation model-free / model-based

## Activations

Choix recommandé pour le premier run:

- [ ] `SwiGLU` comme choix principal
- [ ] `GELU` comme baseline
- [ ] pas de `tanh` ou `sigmoid` comme activation principale

Choix expérimentaux plus tard:

- [ ] tester `Maxout` si on veut explorer une activation plus expressive
- [ ] tester `SIREN` uniquement si on pousse une piste représentation implicite / monde continu

## Efficacité mémoire / compute

- [ ] mixed precision
- [ ] gradient checkpointing
- [ ] séquences / rollouts courts au début
- [ ] taille de latent strictement contrôlée
- [ ] taille MLP contrôlée
- [ ] nombre de heads / layers contrôlé

## Ce qu’on ne fait pas au début

- [ ] pas de long context sophistiqué en v1
- [ ] pas d’activation exotique comme choix par défaut
- [ ] pas de pipeline multi-stage trop complexe avant d’avoir une baseline propre

## MoE

MoE est autorisé en v1, mais seulement si la raison est explicite.

- [ ] Définir pourquoi le MoE aide dans notre cas.
  - plus de capacité totale sous budget actif
  - meilleure spécialisation
  - meilleur ratio perf / compute

- [ ] Définir la version minimale du MoE.
  - nombre total d’experts
  - experts actifs par token / état
  - shared expert ou non
  - router simple top-k

- [ ] Définir le coût réel.
  - surcoût mémoire
  - surcoût routing
  - effet sur stabilité d’entraînement

- [ ] Comparer honnêtement contre une baseline dense de même budget actif

Règle:
- pas de MoE complexe "pour le style"
- seulement si on peut montrer qu’il améliore le ratio capacité / coût dans notre budget

## Décision par défaut si on devait coder demain

- world model de style Dreamer compact
- actor-critic classique dans l’imagination
- benchmark principal `Crafter`
- benchmark de contrôle `DMControl`
- `SwiGLU`
- comparaison `GELU`
- `MoE` simple en v1
- architecture simple avant toute sophistication
