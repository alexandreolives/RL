"""Atomic building blocks."""

from .activations import build_activation
from .config import TransformerConfig, MoEConfig, AttentionConfig, MultimodalConfig, EngramConfig, ByteConfig
from .embeddings import TokenEmbedding, ByteEmbedding, ModalityEmbedding
from .layers import TransformerBlock
from .moe import SparseMoE
from .engram import EngramMemory
from .bytes import BytePatcher
from .residual import FullAttentionResidual
from .contracts import ModularAgentConfig, RouterDecision
from .experiment import RunRecord, parameter_count, write_run_manifest
from .router import RandomRouter, TopKRouter, build_single_expert_router, routing_stats
from .kda import KDA, HybridKDA, GatedMLA, KDAState
from .latent_moe import LatentMoE
from .quantization import BlockFloatFakeQuant, MXFP4FakeQuant, MXFP8FakeQuant, QATLinear
