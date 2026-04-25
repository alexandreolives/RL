# Activations Notes

Objectif de cette note:
- lister les activations réellement utiles
- distinguer ce qui marche en pratique de ce qui est juste élégant
- identifier les vieilles activations sous-utilisées qui restent intéressantes
- garder les papiers importants à portée pour notre objectif `perf max sous budget`

## Verdict court

Si le but est un modèle moderne performant avec un budget compute serré:

1. `SwiGLU`
   meilleur compromis pratique pour LLM / transformer moderne

2. `GELU`
   toujours très solide, simple, stable, bon baseline

3. `SiLU / Swish`
   très bon bloc de base, encore pertinent, surtout historiquement important

4. `Maxout`
   très expressive mais souvent trop lourde en paramètres / compute

5. `Mish`
   souvent bonne empiriquement, mais moins standard et moins "worth it" que SwiGLU en pratique

6. `SIREN / sine activations`
   extrêmement intéressantes, mais surtout pour implicit neural representations, pas comme activation générale de LLM

## Pour notre projet

Si on construit quelque chose de sérieux sous `<= 40GB`:

- baseline pragmatique:
  - `SwiGLU`
- baseline simple et sobre:
  - `GELU`
- branche de recherche plus spéculative:
  - `Maxout` si on veut tester une expressivité plus forte
  - `SIREN` uniquement si on travaille sur une représentation de monde continue / implicite

## Papiers principaux

### SwiGLU / GLU

- Papier:
  - `GLU Variants Improve Transformer`
  - source: `https://arxiv.org/abs/2002.05202`
  - PDF local: [glu_variants_improve_transformer.pdf](../activations/glu_variants_improve_transformer.pdf)

Pourquoi c'est important:
- c'est la référence pratique la plus utile ici
- montre que les variantes GLU, notamment `SwiGLU`, améliorent nettement les transformers
- c'est le point d'ancrage principal si on veut choisir une activation FFN moderne

Conclusion utile:
- si tu veux une seule activation à tester en priorité pour un transformer performant, prends `SwiGLU`

### Swish / SiLU

- Papier:
  - `Searching for Activation Functions`
  - source: `https://arxiv.org/abs/1710.05941`
  - PDF local: [searching_for_activation_functions.pdf](../activations/searching_for_activation_functions.pdf)

Pourquoi c'est important:
- introduit `Swish`
- montre qu'une activation lisse auto-gatée peut battre ReLU dans plusieurs settings

Conclusion utile:
- `Swish/SiLU` a préparé le terrain conceptuel pour `SwiGLU`

### GELU

- Papier:
  - `Gaussian Error Linear Units (GELUs)`
  - source: `https://arxiv.org/abs/1606.08415`
  - PDF local: [gaussian_error_linear_units_gelu.pdf](../activations/gaussian_error_linear_units_gelu.pdf)

Pourquoi c'est important:
- activation standard très crédible
- énormément utilisée dans les transformers
- excellent baseline si on veut un design propre et robuste

Conclusion utile:
- si on veut minimiser le risque de tuning et rester sobre, `GELU` est encore un très bon choix

### tanh

Statut:
- activation historique majeure
- beaucoup plus importante dans les RNN classiques, LSTM, GRU, et dans les réseaux plus anciens
- aujourd'hui rarement le meilleur choix par défaut pour les gros transformers

Pourquoi elle a été importante:
- centrée autour de zéro
- bornée
- plus agréable que la sigmoid pure pour la propagation du signal dans beaucoup de vieux setups

Pourquoi elle a été dépassée:
- saturation dans les grandes amplitudes
- gradients qui disparaissent facilement
- moins favorable à l'entraînement profond et massif que les familles ReLU / GELU / SiLU / GLU

Où elle reste pertinente:
- gates internes de LSTM/GRU
- certaines têtes de sortie bornées
- parfois utile quand on veut explicitement contraindre une représentation dans `[-1, 1]`

Conclusion utile:
- `tanh` n'est pas mauvaise, mais pour notre objectif ce n'est pas la bonne candidate principale
- elle reste surtout une brique de contrôle ou de gating, pas l'activation FFN moderne idéale

### sigmoid

Statut:
- activation fondatrice
- historiquement centrale, mais rarement utilisée seule comme activation cachée principale dans les réseaux profonds modernes

Pourquoi elle a été importante:
- interprétation probabiliste très simple
- sortie dans `[0, 1]`
- très naturelle pour des gates et des probabilités

Pourquoi elle a été dépassée comme activation principale:
- saturation rapide
- fort risque de vanishing gradients
- sortie non centrée en zéro
- en pratique, elle est beaucoup moins compétitive qu'une activation moderne pour des couches cachées profondes

Où elle reste pertinente:
- sortie binaire / Bernoulli
- gates de LSTM / GRU / mécanismes de contrôle
- composante de `SiLU` et de la famille Swish

Conclusion utile:
- `sigmoid` seule n'est pas un bon choix pour notre MLP principal
- en revanche, elle survit de manière très importante à l'intérieur d'autres activations ou des mécanismes de gating

### SiLU / dSiLU en RL

- Papier:
  - `Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning`
  - source: `https://arxiv.org/abs/1702.03118`
  - PDF local: [silu_dsilu_for_rl.pdf](../activations/silu_dsilu_for_rl.pdf)

Pourquoi c'est important:
- directement pertinent pour nous car orienté RL
- montre que `SiLU` et `dSiLU` sont des activations crédibles pour approximation de fonctions en RL

Conclusion utile:
- si on part sur un stack RL compact et qu'on veut sortir du duo ReLU/GELU, c'est une bonne lecture

### Maxout

- Papier:
  - `Maxout Networks`
  - source: `https://arxiv.org/abs/1302.4389`
  - PDF local: [maxout_networks.pdf](../activations/maxout_networks.pdf)

Pourquoi c'est important:
- vieille activation, mais très expressive
- souvent sous-estimée aujourd'hui
- très bon candidat quand on parle de "meilleur en théorie / plus puissant / trop cher"

Pourquoi ce n'est pas partout:
- augmente fortement le coût paramètre / compute
- moins pratique pour des architectures déjà lourdes

Conclusion utile:
- si tu pensais à une vieille activation "pas beaucoup utilisée mais potentiellement top", `Maxout` est l'un des meilleurs candidats

### Mish

- Papier:
  - `Mish: A Self Regularized Non-Monotonic Activation Function`
  - source: `https://arxiv.org/abs/1908.08681`
  - PDF local: [mish_self_regularized_non_monotonic_activation.pdf](../activations/mish_self_regularized_non_monotonic_activation.pdf)

Pourquoi c'est intéressant:
- activation lisse et non monotone
- souvent aimée pour son comportement empirique

Limite pratique:
- pas devenue le standard dominant en transformers
- bénéfice moins "évident" que `SwiGLU` dans les architectures qu'on vise

### PReLU

- Papier:
  - `Delving Deep into Rectifiers`
  - source: `https://arxiv.org/abs/1502.01852`
  - PDF local: [prelu.pdf](../activations/prelu.pdf)

Pourquoi c'est important:
- classique très utile
- montre qu'on peut améliorer ReLU avec une pente apprise

Conclusion utile:
- bon rappel que beaucoup de gains historiques viennent d'ajustements simples, pas de fonctions exotiques

### SReLU

- Papier:
  - `Deep Learning with S-shaped Rectified Linear Activation Units`
  - source: `https://arxiv.org/abs/1512.07030`
  - PDF local: [srelu.pdf](../activations/srelu.pdf)

Pourquoi c'est intéressant:
- plus flexible qu'un simple rectifier
- fait partie des activations "oubliées mais pas idiotes"

Pourquoi ce n'est pas devenu un standard:
- plus de complexité
- moins de traction pratique que GELU / Swish / GLU

### SIREN

- Papier:
  - `Implicit Neural Representations with Periodic Activation Functions`
  - source: `https://arxiv.org/abs/2006.09661`
  - PDF local: [siren.pdf](../activations/siren.pdf)

Pourquoi c'est intéressant:
- très fort sur les représentations implicites continues
- bonne piste si on pense world model comme champ continu ou mémoire spatiale/physique

Pourquoi ce n'est pas une activation généraliste idéale:
- domaine d'excellence spécifique
- réglages et comportement différents d'un LLM/transformer standard

## DELU

J'ai essayé de récupérer un papier DELU, mais le téléchargement automatique a échoué dans ce batch.

Ce qu'il faut retenir:
- `DELU` existe bien dans la littérature comme variante d'ELU
- ce n'est pas un standard moderne pour transformers
- aujourd'hui, si le critère est "meilleure activation pratique", `SwiGLU` reste beaucoup plus importante

## Réponse à la vraie question: y a-t-il mieux que tanh ?

Oui, largement, selon le contexte.

Pour un usage moderne général:
- `tanh` n'est plus le meilleur choix par défaut
- `GELU`, `SiLU/Swish`, `SwiGLU` dominent nettement en pratique sur les gros modèles modernes

Si tu veux "plus fort mais plus lourd":
- `Maxout` est un très bon candidat

Si tu veux "plus exotique mais potentiellement très intéressant":
- `Mish`
- `SIREN` dans les domaines de représentations continues

Et par rapport à `sigmoid`:
- comme activation cachée principale, elle est en général encore moins compétitive que `tanh`
- sa vraie survie moderne est surtout indirecte:
  - `SiLU(x) = x * sigmoid(x)`
  - `Swish` et les variantes GLU exploitent précisément cette idée de gating doux

## Hiérarchie pratique finale

Si on devait vraiment choisir:

1. pour gagner en pratique sur transformer:
   - `SwiGLU`

2. pour un baseline très propre:
   - `GELU`

3. pour RL compact et variantes lisses:
   - `SiLU`

4. pour recherche expressive plus lourde:
   - `Maxout`

5. pour niches représentatives / world modeling continu:
   - `SIREN`

## Ce que je recommande pour nous

Pour la suite du projet:

- piste principale:
  - `SwiGLU`
- baseline secondaire:
  - `GELU`
- branche expérimentale si on veut tester une vieille idée sous-estimée:
  - `Maxout`

Si ton intuition était "il y a une vieille activation qu'on n'a presque jamais utilisée en pratique mais qui pourrait être top", ma meilleure réponse honnête est:

- `Maxout` si on parle d'expressivité générale
- `SIREN` si on parle de représentations implicites continues

## Place exacte de tanh et sigmoid dans notre shortlist

Si je dois les classer pour notre cas:

- `sigmoid`
  - pas candidate principale
  - utile pour les gates, sorties probabilistes, et comme ingrédient de meilleures activations

- `tanh`
  - meilleure que sigmoid comme activation cachée historique
  - mais toujours inférieure en pratique à `GELU`, `SiLU`, `SwiGLU` pour ce qu'on vise

Donc pour notre projet:
- ne pas choisir `sigmoid` comme activation principale
- ne pas choisir `tanh` comme activation principale
- garder `tanh` et `sigmoid` seulement si on a une raison structurelle explicite
