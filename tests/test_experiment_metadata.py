import json

import torch
from torch import nn

from models.atoms.config import TransformerConfig
from models.atoms.experiment import RunRecord, parameter_count, write_run_manifest


def test_parameter_count_and_run_record(tmp_path):
    model = nn.Linear(3, 2)
    assert parameter_count(model) == 8
    model.bias.requires_grad_(False)
    assert parameter_count(model, trainable_only=True) == 6

    path = tmp_path / "run.json"
    RunRecord(seed=1, variant="baseline", parameters=8, trainable_parameters=6).write_json(path)
    assert json.loads(path.read_text())["variant"] == "baseline"


def test_manifest_contains_schema_run_and_config(tmp_path):
    path = tmp_path / "manifest.json"
    write_run_manifest(
        path,
        record=RunRecord(seed=0, variant="baseline", parameters=1, trainable_parameters=1),
        config=TransformerConfig(d_model=8, depth=1, vocab_size=16),
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["run"]["seed"] == 0
    assert payload["config"]["d_model"] == 8
