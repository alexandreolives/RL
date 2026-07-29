# JEPA + Simulus Paper Pack

Source vidéo:
- `https://youtu.be/oM4neOyZOi0`
- titre: `What Is Yann LeCun Cooking? JEPA Explained Simply`

Extraction description faite localement via `yt-dlp`.

## Papiers récupérés

- `v-jepa-2_2506.09985.pdf`
- `i-jepa_2301.08243.pdf`
- `lejepa_2511.08544.pdf`
- `echojepa_2602.02603.pdf`
- `v-jepa_2404.08471.pdf`
- `simulus_m3_2502.11537.pdf`
- `ema_ssl_vit_2104.14294.pdf`
- `simclr_2002.05709.pdf`
- `barlow_twins_2103.03230.pdf`
- `vicreg_2105.04906.pdf`
- `dinov2_2304.07193.pdf`

## Note sur le papier "Original JEPA"

Tentative échouée en direct:
- `https://openreview.net/pdf?id=BZ5a1r-kVsf`
- retour `HTTP 403` depuis cet environnement.

Lien de référence conservé (non téléchargé localement pour le moment):
- `https://openreview.net/pdf?id=BZ5a1r-kVsf`

## Correspondance rapide

- `simulus_m3_2502.11537.pdf`:
  - `M^3: A Modular World Model over Streams of Tokens`
  - modèle RL/world-model associé à Simulus (HF card: `leorc/Simulus`).

## Références ajoutées — prédiction multi-token et diffusion

Ces travaux sont pertinents pour le routeur, les boucles récurrentes et les
experts de diffusion, mais ils ne doivent pas être confondus :

- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) : plusieurs têtes prédisent des tokens futurs depuis un tronc partagé ; le papier étudie aussi la self-speculative decoding et les modèles byte-level.
- [Multi-Token Residual Prediction](https://arxiv.org/abs/2605.18817) : module de prédiction de résidu entre étapes de débruitage pour diffusion de langage, avec modes direct et spéculatif.
- [Fast and Expressive Multi-Token Prediction with Probabilistic Circuits](https://arxiv.org/abs/2511.11346) : distributions jointes de tokens futurs via circuits probabilistes, incluant des structures de type Markov et des variantes compatibles avec la spéculation.
- [On multi-token prediction for efficient LLM inference](https://arxiv.org/abs/2502.09419) : analyse des capacités MTP latentes des modèles entraînés en next-token.
- [GLM-5.2 — Z.ai technical blog](https://z.ai/blog/glm-5.2) : MTP multi-step avec paramètres partagés, partage d'index/KV et rejection sampling ; Z.ai rapporte une hausse de 20 % de la longueur acceptée dans son ablation.
- [Kimi K3 serving notes](https://vllm-project.github.io/vllm/blog/k3.html) : décrit une confidence head qui prédit l'acceptation des tokens proposés et élague les propositions faibles avant vérification.

À ce stade, aucune source primaire consultée ne confirme que le travail
Markov/multi-token ci-dessus provient de Z.ai. L'attribution Z.ai reste donc une
hypothèse à vérifier avant publication. Les deux sources industrielles ci-dessus
confirment toutefois le mécanisme général « proposer plusieurs tokens, estimer
la confiance/acceptation, arrêter ou vérifier tôt ».
