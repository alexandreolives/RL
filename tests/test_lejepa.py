from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from eval.transformer.lejepa import lejepa_loss, make_masked_views, sigreg_loss


def test_masked_views_are_created_before_encoding_and_deterministic() -> None:
    ids = torch.arange(24).view(3, 8)
    views_a = make_masked_views(ids, num_views=2, mask_ratio=0.5, mask_token_id=256, seed=17)
    views_b = make_masked_views(ids, num_views=2, mask_ratio=0.5, mask_token_id=256, seed=17)

    assert len(views_a) == 2
    assert all(torch.equal(a, b) for a, b in zip(views_a, views_b, strict=True))
    assert not torch.equal(views_a[0], views_a[1])
    for view in views_a:
        masked = view.eq(256)
        assert masked.any(dim=1).all()
        assert (~masked).any(dim=1).all()
        assert torch.equal(view[~masked], ids[~masked])


def test_sigreg_prefers_an_isotropic_gaussian_to_collapse() -> None:
    generator = torch.Generator().manual_seed(5)
    gaussian = torch.randn((4096, 16), generator=generator)
    collapsed = torch.zeros_like(gaussian)

    gaussian_loss = sigreg_loss(gaussian, global_step=11, num_slices=128)
    collapsed_loss = sigreg_loss(collapsed, global_step=11, num_slices=128)

    assert gaussian_loss < collapsed_loss


def test_lejepa_matches_algorithm_two_and_backpropagates_through_every_view() -> None:
    view_a = torch.randn(32, 12, requires_grad=True)
    view_b = torch.randn(32, 12, requires_grad=True)
    result = lejepa_loss(
        [view_a, view_b],
        lambd=0.2,
        global_step=3,
        num_slices=32,
        num_knots=17,
    )

    expected = 0.8 * result.prediction + 0.2 * result.sigreg
    assert torch.allclose(result.total, expected)
    result.total.backward()
    assert view_a.grad is not None and torch.isfinite(view_a.grad).all()
    assert view_b.grad is not None and torch.isfinite(view_b.grad).all()
    assert view_a.grad.abs().sum() > 0
    assert view_b.grad.abs().sum() > 0


class _RecordingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_byte_first=True,
            bytes=SimpleNamespace(patch_size=1),
        )
        self.embedding = nn.Embedding(260, 8)
        self.head = nn.Linear(8, 260)
        self.inputs: list[torch.Tensor] = []

    def forward(self, *, byte_ids: torch.Tensor, return_hidden: bool):
        self.inputs.append(byte_ids.detach().clone())
        hidden = self.embedding(byte_ids)
        return self.head(hidden), hidden


def test_text_lm_step_uses_separate_corrupted_forwards_for_real_lejepa() -> None:
    # Importing here keeps the standalone LeJEPA module usable independently
    # from the optional dataset runner.
    from eval.transformer.train_text_lm_compare import step_loss

    model = _RecordingEncoder()
    batch = torch.randint(0, 256, (4, 9))
    lm_loss, ssl_loss, total = step_loss(
        model,
        batch,
        jepa_mode="lejepa",
        jepa_weight=0.05,
        jepa_mask_ratio=0.4,
        lejepa_lambda=0.1,
        lejepa_num_views=2,
        lejepa_num_slices=16,
        view_seed=29,
    )

    assert len(model.inputs) == 3
    assert torch.equal(model.inputs[0], batch[:, :-1])
    assert all(view.eq(256).any() for view in model.inputs[1:])
    assert torch.isfinite(lm_loss) and torch.isfinite(ssl_loss)
    assert torch.allclose(total, lm_loss + 0.05 * ssl_loss)
    total.backward()
    assert model.embedding.weight.grad is not None
