import json

import torch
from torch import nn

from models.atoms.experiment import RunRecord, parameter_count


def test_parameter_count_and_run_record(tmp_path):
    model = nn.Linear(3, 2)
    assert parameter_count(model) == 8
    model.bias.requires_grad_(False)
    assert parameter_count(model, trainable_only=True) == 6

    path = tmp_path / "run.json"
    RunRecord(seed=1, variant="baseline", parameters=8, trainable_parameters=6).write_json(path)
    assert json.loads(path.read_text())["variant"] == "baseline"
