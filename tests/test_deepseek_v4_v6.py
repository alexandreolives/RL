from __future__ import annotations

import unittest

import torch

from models.example import apply_model_size, build_deepseek_v4_v6_config, build_model
from models.molecules import DeepseekV4ReferenceCache, DeepseekV4ReferenceMolecule


def build_tiny_v6() -> DeepseekV4ReferenceMolecule:
    config = apply_model_size(
        build_deepseek_v4_v6_config(input_mode="symbolic", attention_backend="eager"),
        "tiny",
        input_mode="symbolic",
    )
    config.depth = 4
    config.max_seq_len = 256
    config.attention.sliding_window = 8
    config.attention.compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 8,
    }
    config.attention.index_topk = 2
    config.moe.num_experts = 4
    config.moe.top_k = 2
    config.hc_mult = 2
    config.mhc_sinkhorn_iters = 4
    return build_model(config)


class DeepseekV4V6Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = build_tiny_v6().eval()

    def test_uses_official_layer_schedule_and_balanced_hash_routes(self) -> None:
        self.assertIsInstance(self.model, DeepseekV4ReferenceMolecule)
        self.assertEqual(
            self.model.hf_config.layer_types,
            [
                "heavily_compressed_attention",
                "heavily_compressed_attention",
                "heavily_compressed_attention",
                "compressed_sparse_attention",
            ],
        )
        table = self.model.hf_model.model.layers[0].mlp.gate.tid2eid
        self.assertTrue(torch.all(table[:, 0] != table[:, 1]))

    def test_full_forward_matches_incremental_cache(self) -> None:
        input_ids = torch.randint(0, 260, (1, 16))
        with torch.no_grad():
            full = self.model(token_ids=input_ids, use_cache=False)
            cache = None
            pieces = []
            for position in range(input_ids.size(1)):
                logits, cache = self.model(
                    token_ids=input_ids[:, position : position + 1],
                    past_key_values=cache,
                    return_cache=True,
                )
                pieces.append(logits)
        incremental = torch.cat(pieces, dim=1)
        torch.testing.assert_close(full, incremental, atol=2e-5, rtol=2e-5)
        self.assertEqual(cache.get_seq_length(), input_ids.size(1))

    def test_sliding_branch_is_live_during_compressor_warmup(self) -> None:
        captured = {}
        attention = self.model.hf_model.model.layers[0].self_attn

        def capture_grouped_input(_module, args) -> None:
            captured["attention"] = args[0].detach()

        handle = attention.o_a_proj.register_forward_pre_hook(capture_grouped_input)
        try:
            with torch.no_grad():
                self.model(token_ids=torch.randint(0, 260, (1, 16)), use_cache=False)
        finally:
            handle.remove()

        early_norms = captured["attention"][:, :7].flatten(2).norm(dim=-1)
        self.assertTrue(torch.all(early_norms > 0), early_norms)

    def test_cache_clone_and_reset_cover_compressor_state(self) -> None:
        cache = self.model.new_cache()
        with torch.no_grad():
            _, cache = self.model(
                token_ids=torch.randint(0, 260, (1, 16)),
                past_key_values=cache,
                return_cache=True,
            )
        self.assertIsInstance(cache, DeepseekV4ReferenceCache)
        clone = cache.clone()

        for source, copied in zip(cache.layers, clone.layers):
            self.assertIs(source.keys, source.values)
            source_counts = getattr(source, "entry_count", {})
            copied_counts = getattr(copied, "entry_count", {})
            self.assertEqual(source_counts, copied_counts)
            for name, tensor in getattr(source, "compressed_kv", {}).items():
                copied_tensor = copied.compressed_kv[name]
                if tensor is not None:
                    self.assertIsNot(tensor, copied_tensor)
                    torch.testing.assert_close(tensor, copied_tensor)

        cache.reset()
        self.assertEqual(cache.get_seq_length(), 0)
        for layer in cache.layers:
            self.assertTrue(all(value == 0 for value in getattr(layer, "entry_count", {}).values()))
            for state_name in ("buffer_kv", "buffer_gate", "compressed_kv", "overlap_kv", "overlap_gate"):
                state = getattr(layer, state_name, {})
                self.assertTrue(all(value is None for value in state.values()))

    def test_training_backward_reaches_attention_and_sparse_experts(self) -> None:
        self.model.train()
        input_ids = torch.randint(0, 260, (2, 8))
        logits = self.model(token_ids=input_ids, use_cache=False)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 260), input_ids[:, 1:].reshape(-1))
        loss.backward()

        attention_grad = self.model.hf_model.model.layers[0].self_attn.q_a_proj.weight.grad
        expert_grad = self.model.hf_model.model.layers[-1].mlp.experts.gate_up_proj.grad
        self.assertIsNotNone(attention_grad)
        self.assertIsNotNone(expert_grad)
        self.assertTrue(torch.isfinite(attention_grad).all())
        self.assertTrue(torch.isfinite(expert_grad).all())


if __name__ == "__main__":
    unittest.main()
