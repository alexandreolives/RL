# A Closer Look At Kimi K3's INSANE Architecture Breakthrough — source archive

- Video: https://www.youtube.com/watch?v=g683I1-4MKE
- Channel: bycloud
- Published: 2026-07-29
- Retrieved: 2026-07-30
- Primary reference: https://arxiv.org/abs/2607.24653
- Official implementation: https://github.com/MoonshotAI/Kimi-K3
- Expert-parallel runtime: https://github.com/MoonshotAI/MoonEP

## Archived material

- [`g683I1-4MKE.en.vtt`](g683I1-4MKE.en.vtt): generated English transcript.
- [`g683I1-4MKE.en-orig.vtt`](g683I1-4MKE.en-orig.vtt): alternate subtitle
  track returned by YouTube.
- [`g683I1-4MKE.description`](g683I1-4MKE.description): original description.
- [`g683I1-4MKE.info.json`](g683I1-4MKE.info.json): yt-dlp metadata.

## Ideas relevant to this repository

1. **Kimi Delta Attention (KDA).** A recurrent linear-attention state updated
   with the delta rule and channel-wise decay. It is a candidate replacement
   for the fast path in our recurrent Loop, but it must retain a full-attention
   fallback for exact retrieval.
2. **Hybrid attention schedule.** The video reports a KDA/full MLA ratio near
   `3:1`. Test KDA/FFT/SSM fast layers with a final or periodic global-attention
   layer, measuring quality, KV/state memory and latency at equal active FLOPs.
3. **Attention Residuals.** Replace fixed additive depth residuals with
   content-dependent softmax aggregation over previous layer outputs. This is
   directly relevant to our loop depth and LayerScale experiments.
4. **Stable LatentMoE.** Compress the latent before expert dispatch and expand
   it after routing, with quantile/load balancing. This is more actionable for
   our MoE roadmap than naïve token-to-expert all-to-all dispatch.
5. **Dynamic redundant experts / MoonEP.** Replicate overloaded experts at
   runtime while preserving balanced per-rank token counts. This belongs to
   the systems track, not the model-quality ablation.
6. **Deployment-aware training.** Test MXFP4 expert weights, MXFP8 activations,
   QAT and Eagle-style speculative heads only after FP/BF16 controls are fixed.
7. **Native multimodal training and long-context curriculum.** Evaluate joint
   vision-language training, progressive context lengths, and scattered-memory
   synthetic tasks rather than attaching vision after language pretraining.
8. **Multi-teacher/post-training mixture.** Add specialized coding, agent and
   reasoning-effort teachers as a later post-training experiment; do not mix
   this with the architecture ablations.

The video is explanatory secondary evidence. Architecture and numerical claims
must be checked against the Kimi K3 report and Kimi Linear paper before being
used as scientific results.
