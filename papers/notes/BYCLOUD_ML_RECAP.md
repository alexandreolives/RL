# ByCloud ML Recap

Notes de synthèse à partir des vidéos déjà vues sur la chaîne `@bycloudAI`.

Base de travail utilisée:
- descriptions, métadonnées et sous-titres récupérés via `yt-dlp`
- fichiers stockés dans `.cache/bycloud/`
- ce document résume les points avancés dans les vidéos; ce n'est pas une validation indépendante de chaque papier

## Vidéos couvertes

1. Google's TurboQuant Memory Reduction Claim vs Reality
   URL: https://www.youtube.com/watch?v=haoAI2lIZ74
2. DeepSeek's Insane Architecture Breakthrough [Engram Explained]
   URL: https://www.youtube.com/watch?v=xUlX6jvwVfM
3. The Implosion of the Top Open Source Lab Qwen
   URL: https://www.youtube.com/watch?v=aQr_FWJETOk
4. DeepSeek Just Added Parameters Where There Were None
   URL: https://www.youtube.com/watch?v=Gb0jLRC8uWM
5. DeepSeek V3.2 Just Broke SoTA Again… But How?
   URL: https://www.youtube.com/watch?v=pljoUcBniPQ
6. How did a 27M Model even beat ChatGPT?
   URL: https://www.youtube.com/watch?v=ZgwHaI2C-9s
7. POV: Chinese AI Lab Teaching Everyone How To Save Millions of Dollars
   URL: https://www.youtube.com/watch?v=cJeqGq0Bx1M
8. What If We Remove Tokenization In LLMs?
   URL: https://www.youtube.com/watch?v=G7ryb91BDG8
9. 1-Bit LLM: The Most Efficient LLM Possible?
   URL: https://www.youtube.com/watch?v=7hMoz9q4zv0
10. The REAL AI Architecture That Unifies Vision & Language
   URL: https://www.youtube.com/watch?v=obSNYqgL53k
11. How DeepSeek Built The Current "Best" Math Prover AI
   URL: https://www.youtube.com/watch?v=vhXDKif9mPU
12. AI News: DeepSeek-R1 V2, the new open source SoTA!
   URL: https://www.youtube.com/watch?v=_5Xv3kXyBDE

## Ce qui revient souvent et vaut la peine d'être retenu

- Les gains les plus crédibles ne viennent pas seulement du scale, mais d'un meilleur usage du compute: sparsity, mémoire conditionnelle, quantization, data synthesis, retry budget, organisation infra.
- Beaucoup de résultats impressionnants s'écroulent ou deviennent moins impressionnants dès qu'on remet le bon baseline: pas de comparaison FP32 inutile, pas de benchmark trop étroit, pas de mélange entre modèle généraliste et solveur spécialisé.
- Les papiers architecture solides sont ceux qui enlèvent les variables de confusion et documentent les coûts réels: overhead mémoire, budget d'entraînement, retries, coût d'inférence et pas seulement score final.
- Les pipelines avec vérificateur externe ou structure de tâche explicite restent extrêmement puissants: theorem proving, recursive decomposition, byte grouping, model merging en pré-training.
- Le "open weights" est utile mais ne veut pas dire "facile à faire tourner". Plusieurs vidéos insistent sur ce décalage entre disponibilité et exploitabilité.

## Notes utiles par vidéo

### 1. TurboQuant

Pourquoi c'est intéressant:
- Le sujet est important parce que la KV cache devient un vrai bottleneck mémoire à long contexte.
- Le point à retenir n'est pas seulement "compression", mais "compression qui doit être comparée aux vrais baselines déjà utilisés en prod".

Critique méthodologique à garder:
- ByCloud insiste que le claim `up to 8x speedup` est marketing car comparé à un baseline 32-bit non quantifié que presque personne n'utilise en pratique.
- La vraie question devrait être: qu'est-ce que TurboQuant apporte par rapport aux méthodes déjà réalistes en 4-bit / low-bit KV cache.
- Le message important pour plus tard: toujours exiger la comparaison contre la stack réellement déployée, pas contre un strawman.

Papiers / liens:
- TurboQuant: https://arxiv.org/abs/2504.19874
- OpenReview comments: https://openreview.net/forum?id=tO3ASKZlok
- PolarQuant: https://arxiv.org/abs/2502.02617
- QJL: https://arxiv.org/abs/2406.03482
- KIVI: https://arxiv.org/abs/2402.02750
- RabitQ: https://arxiv.org/abs/2405.12497

### 2. Engram

Pourquoi c'est intéressant:
- Idée d'ajouter un troisième bloc au transformer, à côté de l'attention et du FFN, pour faire de la mémoire conditionnelle via lookup scalable.
- Le bénéfice conceptuel est fort: éviter de reconstruire encore et encore des patterns multi-tokens déjà vus.

Point à retenir:
- Si une structure revient souvent, il peut être rentable de la traiter comme une unité mémorisée plutôt que de refaire tout le calcul compositionnel à chaque occurrence.
- C'est une autre forme de conditional compute / conditional memory.

Critique méthodologique à garder:
- La vidéo est surtout positive ici: elle souligne que le papier enlève beaucoup de variables de confusion et pousse les ablations beaucoup plus loin que la moyenne.
- Le méta-point réutilisable est donc: lorsqu'on évalue une nouvelle brique d'architecture, la qualité des ablations compte autant que le score final.

Papiers / liens:
- Engram: https://arxiv.org/abs/2601.07372v1
- DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
- DeepSeek sparse attention report: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

### 3. Qwen implosion

Pourquoi c'est intéressant:
- Ce n'est pas une vidéo papier, mais une vidéo importante sur l'organisation de la recherche.
- Le point fort: l'infra, le pre-training, le post-training et le model-making doivent être alignés étroitement.

Point à retenir:
- Une équipe modèle sans contrôle fort sur l'infrastructure perd en vitesse d'itération et en efficacité.
- Pour des labs frontier, l'organisation est une variable technique, pas juste RH.

Critique à garder:
- La vidéo avance surtout une lecture organisationnelle: la fragmentation des équipes nuit à la boucle R&D.
- Pour plus tard, il faut traiter la structure d'équipe comme un multiplicateur de qualité de recherche et de coût de training.

### 4. mHC / Hyper-Connections

Pourquoi c'est intéressant:
- DeepSeek remet en cause la forme canonique du residual path en mettant plusieurs streams parallèles avec des poids de mélange appris.
- Idée simple, effet potentiellement large.

Point à retenir:
- Les skip connections peuvent être enrichies sans exploser le coût si l'implémentation limite surtout les memory stalls.
- Le bénéfice évoqué: gains cohérents sur tailles et budgets d'entraînement différents.

Critique méthodologique à garder:
- L'objection naturelle est l'overhead compute. La vidéo dit que le vrai coût n'est pas "4x compute", mais surtout l'organisation du trafic mémoire.
- Le point pratique: mesurer les coûts réels au niveau système, pas juste le FLOP count théorique.

Papiers / liens:
- ByteDance Hyper Connections: https://arxiv.org/abs/2409.19606
- DeepSeek mHC: https://arxiv.org/abs/2512.24880

### 5. DeepSeek V3.2

Pourquoi c'est intéressant:
- Cas d'école d'un frontier model open-weights très compétitif avec coût d'inférence présenté comme exceptionnellement bas.
- La vidéo insiste sur l'association entre bonnes idées de recherche et scale, pas sur un simple "more GPUs".

Point à retenir:
- Séparer un modèle généraliste d'une variante "extended reasoning" permet de distinguer usage pratique et plafond de performance.
- Le fait qu'un modèle "special" surperforme ne dit pas automatiquement que le modèle principal est meilleur sur tous les cas d'usage.

Critique méthodologique à garder:
- ByCloud précise que le modèle qui bat les références est la variante `Speciale`, pas le `3.2` de base.
- Les comparaisons doivent donc préciser quelle version, quel budget token et quel type de tâche sont utilisés.
- La vidéo note aussi que le coding reste légèrement en retrait de certains concurrents malgré de très bons scores en maths.

Liens:
- DeepSeek-V3.2: https://rebrand.ly/s6apdew
- HF V3.2: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- HF V3.2-Speciale: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale
- DeepSeek-V3.2-exp report: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

### 6. HRM / TRM / "27M beats ChatGPT"

Pourquoi c'est intéressant:
- Très bon rappel qu'un benchmark peut dramatiquement tromper l'intuition publique.
- Les modèles récursifs ou à boucle interne peuvent allouer plus de compute aux parties difficiles sans devenir de gros LLMs généralistes.

Point à retenir:
- L'architecture ou la dynamique d'inférence peut compter plus que la taille brute, mais souvent dans un domaine étroit.
- Le pattern général intéressant est le test-time compute structuré et spécialisé.

Critique méthodologique à garder:
- La vidéo dit explicitement que le titre "beat ChatGPT" est trompeur si on compare un solveur spécialisé ARC-AGI à un LLM généraliste.
- ARC-AGI donne l'illusion que tous les modèles du leaderboard ont des capacités comparables, ce qui est faux.
- Le modèle est très bon sur un cadre de puzzles logique très défini, pas sur le langage général.

Papiers / liens:
- HRM: https://arxiv.org/abs/2506.21734
- TRM: https://arxiv.org/abs/2510.04871

### 7. ByteDance Seed / model merging en pré-training

Pourquoi c'est intéressant:
- Très bon sujet "training economics": utiliser le model merging comme estimateur ou accélérateur dans la phase de pré-training / annealing.
- L'idée utile est de projeter la performance d'un modèle annealed sans payer toute la fin du run.

Point à retenir:
- Les checkpoints intermédiaires contiennent de l'information utile qu'on peut agréger intelligemment.
- Le merge peut servir d'outil de pilotage budgétaire, pas seulement d'outil de post-training.

Critique méthodologique à garder:
- La vidéo souligne que les expériences sont coûteuses, donc rares, et que cela limite la qualité de validation externe.
- Réserve importante: il faudrait tester davantage de tokens d'annealing pour savoir si l'avantage du merge tient sur plus long horizon.
- Bonne question laissée ouverte: le merge évite-t-il réellement du compute final ou seulement du compute court terme avant rattrapage par annealing.

Papiers / liens:
- Model Merging in Pre-training of Large Language Models: https://alphaxiv.org/abs/2505.12082

### 8. Supprimer la tokenization

Pourquoi c'est intéressant:
- Sujet fondamental: la tokenization est à la fois un artifice pratique et une source de limitations structurelles.
- La perspective byte-level / patch-level force à repenser l'allocation du compute.

Point à retenir:
- Les tokens sur-segmentent certaines langues, gèrent mal les typos, et dépensent autant de compute sur ponctuation que sur contenu dense.
- Les modèles type BLT cherchent à faire du compute adaptatif selon l'information réelle du flux brut.

Critique méthodologique à garder:
- La vidéo rappelle qu'un tokenizer est une étape séparée entraînée à part, souvent biaisée vers l'anglais.
- Elle laisse aussi ouverte la question du coût et des collisions / patch definitions lorsqu'on passe au byte-level.
- Le point important n'est pas "plus pur = meilleur", mais "peut-on allouer le compute de façon plus sémantique que le simple token".

Papiers / liens:
- BLT: https://arxiv.org/abs/2412.09871
- Code BLT: https://github.com/facebookresearch/blt
- From Bytes to Ideas: https://arxiv.org/abs/2506.14761

### 9. 1-bit LLM / BitNet

Pourquoi c'est intéressant:
- Sujet central pour rendre les gros modèles réellement exécutables hors datacenter.
- La bonne intuition: quantization extrême n'est pas qu'une compression après coup, ça peut être un régime d'entraînement natif.

Point à retenir:
- Les gains dépendent autant du hardware path et des kernels que de la théorie de quantization.
- Le message réutilisable: un modèle low-bit bien entraîné peut être préférable à un plus petit modèle dense.

Critique méthodologique à garder:
- La vidéo fait bien la distinction entre post-training quantization et entraînement natif 1-bit / 1.58-bit.
- Réserve importante: les gros speedups annoncés ne doivent pas être lus indépendamment de l'implémentation système.
- Le 1-bit pur a des limites de représentation; le passage au 1.58-bit sert précisément à corriger cette rigidité.

Papiers / liens:
- Quantifying Capabilities Across Scale and Precision: https://arxiv.org/abs/2405.03146v2
- BitNet: https://arxiv.org/abs/2310.11453v1
- 1.58-bit LLMs: https://arxiv.org/abs/2402.17764v1
- BitNet a4.8: https://arxiv.org/abs/2411.04965v1
- BitNet b1.58 2B4T: https://arxiv.org/abs/2504.12285
- BitNet code: https://github.com/microsoft/BitNet

### 10. Early fusion multimodal

Pourquoi c'est intéressant:
- Très bonne vidéo sur la différence entre empiler des spécialistes et entraîner un vrai modèle multimodal unifié.
- Le message utile pour plus tard: une architecture unifiée peut émerger avec des spécialisations internes sans imposer des modules séparés.

Point à retenir:
- Early fusion permet l'apprentissage end-to-end de représentations jointes vision-langage.
- La spécialisation peut émerger naturellement dans les experts au lieu d'être codée à la main.

Critique méthodologique à garder:
- La vidéo insiste que le consensus "late fusion = meilleur grâce aux spécialistes pré-entraînés" n'est pas garanti.
- En revanche, elle garde une réserve sur le fait de tokeniser l'image elle-même; ce n'est peut-être pas la fin de l'histoire architecturale.
- Autre contrainte persistante: le coût contextuel des images haute résolution.

Papiers / liens:
- Chameleon: https://arxiv.org/abs/2405.09818
- Scaling Laws for Native Multimodal Models: https://www.arxiv.org/abs/2504.07951
- Scaling Pre-training to One Hundred Billion Data for Vision Language Models: https://arxiv.org/abs/2502.07617

### 11. DeepSeek Prover V2

Pourquoi c'est intéressant:
- Très bon exemple de pipeline où la structure de tâche et le vérificateur externe dominent la qualité.
- La combinaison `generalist LLM -> decomposition -> Lean proofs -> verifier -> RL` est un pattern très réutilisable.

Point à retenir:
- Pour les domaines rigoureux, il vaut mieux synthétiser des données validées par exécution / vérification que pousser du CoT vague.
- La décomposition récursive en sous-buts produit des étapes utiles au lieu de "thinking tokens" arbitraires.

Critique méthodologique à garder:
- La vidéo rappelle que le score impressionnant sur Putnam Bench est obtenu avec jusqu'à 1024 essais.
- Le bon réflexe est donc de comparer les modèles à retry budget égal.
- Autre point fort: le gros modèle n'écrit pas la donnée à la main; un petit prover moins coûteux sert à construire le dataset.

Papiers / liens:
- DeepSeek-Prover-V2: https://arxiv.org/abs/2504.21801
- DeepSeek V3 report: https://arxiv.org/abs/2412.19437
- Kimina Prover: https://arxiv.org/abs/2504.11354

### 12. R1 V2 news

Pourquoi c'est intéressant:
- Bon signal de tendance produit, mais plus "news + benchmark scan" que vrai deep dive papier.
- Utile pour suivre l'évolution de l'écosystème open-source et la distillation de reasoning traces.

Point à retenir:
- Le delta de performance d'une révision de modèle peut être significatif même si le lab parle de "minor upgrade".
- Les distillations vers de petits modèles restent importantes pour diffusion pratique.

Critique méthodologique à garder:
- La vidéo note explicitement que certains résultats restent "weird", surtout en coding / physics, malgré de meilleurs benchmarks.
- Le processus de pensée affiché par le modèle est jugé "questionable", donc ne pas confondre score, style de sortie et robustesse réelle.

Liens:
- Chatbot: https://chat.deepseek.com/
- HF DeepSeek-R1-0528: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
- HF distilled Qwen3 8B: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

## Résumé actionnable pour du ML futur

Si je devais condenser tout ça en priorités de recherche / produit:

1. Toujours auditer les baselines et les budgets réels.
   Les trois pièges les plus visibles ici sont:
   baseline irréaliste, benchmark trop étroit, retry budget caché.

2. Regarder les briques qui réallouent le compute.
   Les meilleurs fils rouges de cette sélection sont:
   sparse attention, mémoire conditionnelle, hyper-connections, byte grouping, recursive loops, theorem-proving with verifier.

3. Séparer performance "headline" et utilité réelle.
   Open weights, top benchmark, ou "beats X" ne disent pas:
   coût réel, généricité, stabilité, capacité de déploiement.

4. Ne pas sous-estimer l'ingénierie système et l'organisation.
   Plusieurs vidéos convergent sur un même fait:
   infra, memory traffic, kernels, équipe training/inference, structure de lab.

5. Pour des domaines formels, préférer les pipelines vérifiables.
   Si le domaine accepte un exécuteur ou un vérificateur externe,
   il faut probablement structurer l'apprentissage autour de lui.

## Où retrouver les artefacts

- Dossier local: `.cache/bycloud/`
- Fichiers disponibles par vidéo:
  - `*.info.json`
  - `*.description`
  - `*.en.vtt`
  - `*.en-orig.vtt`

## Papiers cibles pour la suite

Ces papiers ne viennent pas de la liste ByCloud ci-dessus; ils sont ajoutés pour cadrer l'objectif du repo:
faire du RL / world modeling très performant sous contrainte de compute et de mémoire.

### M^3

- Titre: `M^3: A Modular World Model over Streams of Tokens`
- URL: `https://arxiv.org/abs/2502.11537`
- Pourquoi c'est pertinent:
  - world model modulaire sur flux de tokens
  - vise une bonne sample efficiency pour les world models sans planning explicite
  - bon candidat si on veut composer observation/action modalities sans système trop monolithique

### Dreamer 4

- Titre: `Dreamer 4: Training Agents Inside of Scalable World Models`
- URL: `https://arxiv.org/abs/2509.24527`
- Pourquoi c'est pertinent:
  - point de référence récent pour RL dans l'imagination avec world model scalable
  - insiste sur l'entraînement d'agents dans un simulateur appris rapide et précis
  - important pour savoir ce qui est vraiment "haut niveau" aujourd'hui en world-model RL

### LeWorldModel

- Titre: `LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels`
- URL: `https://arxiv.org/abs/2603.19312`
- Pourquoi c'est pertinent:
  - colle bien à l'objectif petit budget: le papier annonce un entraînement sur un seul GPU en quelques heures
  - JEPA end-to-end depuis pixels avec régularisation gaussienne des latents pour éviter le collapse
  - très intéressant si on veut des représentations stables, compactes et utiles pour contrôle / planning
