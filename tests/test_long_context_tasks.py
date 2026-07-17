from __future__ import annotations

import torch

from eval.transformer.long_context_accuracy import (
    ANSWER_MARKER,
    ASSIGN_MARKER,
    QUERY_MARKER,
    TRACK_MARKER,
    make_variable_tracking_token_batch,
    summarize_logits,
)


def test_variable_tracking_batch_encodes_a_valid_copy_chain() -> None:
    seq_len = 60
    ids, modality_ids, answers = make_variable_tracking_token_batch(
        batch=8, seq_len=seq_len, device=torch.device("cpu")
    )
    assert modality_ids is None
    for row, answer in zip(ids, answers):
        p1, p2, p3 = seq_len // 6, 2 * seq_len // 6, 3 * seq_len // 6
        assert row[p1].item() == ASSIGN_MARKER
        assert row[p1 + 2].item() == answer.item()
        assert row[p1 + 3].item() == ANSWER_MARKER
        assert row[p2 + 2].item() == row[p1 + 1].item()
        assert row[p3 + 2].item() == row[p2 + 1].item()
        assert row[-4].item() == TRACK_MARKER
        assert row[-3].item() == row[p3 + 1].item()
        assert row[-2].item() == QUERY_MARKER


def test_task_metrics_report_perfect_oracle_predictions() -> None:
    answers = torch.tensor([3, 7, 11])
    logits = torch.zeros(3, 260)
    logits[torch.arange(3), answers] = 20
    metrics = summarize_logits(logits, answers, vocab_size=260)
    assert metrics["accuracy"] == 1.0
    assert metrics["target_rank"] == 1.0
    assert metrics["target_prob"] > 0.99
