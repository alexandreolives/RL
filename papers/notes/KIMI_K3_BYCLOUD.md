# Kimi K3 — inspirations for our hybrid architecture

Source: [ByCloud video](https://www.youtube.com/watch?v=g683I1-4MKE), archived
under `youtube/bycloud/g683I1-4MKE_A Closer Look At Kimi K3s INSANE Architecture Breakthrough/`.

Primary sources:

- [Kimi K3 technical report](https://arxiv.org/abs/2607.24653)
- [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [LatentMoE](https://arxiv.org/abs/2601.18089)
- [MoonEP](https://github.com/MoonshotAI/MoonEP)

## Architecture implications and experiments for our stack

This note is not a reproduction plan for Kimi K3. The useful question for this
repository is which ideas compose well with our existing Loop, Engram, JEPA,
spectral path, dynamic router and LoRA experts. Each item below therefore has
to be tested both alone and in combinations under matched active compute.

KDA compresses the sequence into a fixed recurrent state and applies a
delta-rule correction rather than blindly accumulating key/value pairs. Its
channel-wise decay gives different latent channels different memory lifetimes.
This matches our recurrent-memory objective, but exact retrieval remains a
known weakness; the proposed experiment is therefore a KDA+Loop/Engram/JEPA
hybrid fast/slow stack, not a pure replacement for global attention.

The reported hybrid schedule is approximately three KDA layers for one full
MLA layer. We should test ratios `3:1`, `1:1`, and attention-only under equal
active FLOPs and report long-context retrieval, state/KV memory, p50/p95
latency, and transfer to multimodal streams.

Attention Residuals is depth-wise attention over prior layer outputs. It is
orthogonal to token attention and directly complements our shared Loop and
LayerScale. The implementation must compare standard residual, AttnRes,
block-AttnRes, LayerScale, and combinations at matched depth.

Stable LatentMoE reduces expert-parallel communication by routing a compressed
latent representation, while balancing expert load. Add it after the current
typed top-k router; measure dispatch bytes and quality separately from routing
accuracy. MoonEP's dynamic redundant experts are a runtime load-balancing
mechanism and should be evaluated independently from the learned router.

K3's deployment ideas (MXFP4/MXFP8 QAT, speculative Eagle-style decoding,
progressive 8K→64K→256K→1M context training, native vision training) are
roadmap experiments, not assumptions about our current small agent.
