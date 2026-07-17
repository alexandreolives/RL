import torch

from models.atoms.activations import (
    ScheduledSquaredActivation,
    StochasticScheduledSquaredActivation,
    build_activation,
    set_stochastic_squared_branch,
)
from eval.transformer.train_long_context_compare import build_train_model


def test_squared_activation_endpoints_and_gradients():
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    relu = build_activation("squared_relu")(x)
    gelu = build_activation("squared_gelu")(x)
    assert torch.all(relu >= 0)
    assert torch.all(gelu >= 0)
    (relu.sum() + gelu.sum()).backward()
    assert x.grad is not None


def test_squared_schedule_endpoints():
    x = torch.linspace(-2, 2, 17)
    schedule = ScheduledSquaredActivation()
    relu = build_activation("squared_relu")(x)
    gelu = build_activation("squared_gelu")(x)
    schedule.set_alpha(0)
    assert torch.allclose(schedule(x), relu)
    schedule.set_alpha(1)
    assert torch.allclose(schedule(x), gelu)


def test_squared_schedule_supports_bfloat16():
    x = torch.linspace(-2, 2, 17, dtype=torch.bfloat16, requires_grad=True)
    schedule = ScheduledSquaredActivation(alpha=0.5)
    y = schedule(x)
    assert y.dtype == torch.bfloat16
    y.sum().backward()
    assert x.grad is not None


def test_long_context_training_model_uses_requested_activation():
    model = build_train_model(
        "engram_noconv",
        torch.device("cpu"),
        model_size="tiny",
        input_mode="symbolic",
        attention_backend="eager",
        activation="squared_schedule",
    )
    scheduled = [m for m in model.modules() if isinstance(m, ScheduledSquaredActivation)]
    assert scheduled


def test_stochastic_schedule_uses_single_deterministic_eval_branch():
    x = torch.linspace(-2, 2, 17)
    schedule = StochasticScheduledSquaredActivation(alpha=0.25).eval()
    assert torch.allclose(schedule(x), build_activation("squared_relu")(x))
    schedule.set_alpha(0.75)
    assert torch.allclose(schedule(x), build_activation("squared_gelu")(x))


def test_stochastic_schedule_training_reaches_both_endpoints():
    x = torch.linspace(-2, 2, 17)
    schedule = StochasticScheduledSquaredActivation(alpha=0.0).train()
    assert torch.allclose(schedule(x), build_activation("squared_relu")(x))
    schedule.set_alpha(1.0)
    assert torch.allclose(schedule(x), build_activation("squared_gelu")(x))


def test_stochastic_schedule_is_hot_swapped_off_forward_path():
    model = build_train_model(
        "engram_noconv",
        torch.device("cpu"),
        model_size="tiny",
        input_mode="symbolic",
        attention_backend="eager",
        activation="squared_stochastic_schedule",
    )
    count = set_stochastic_squared_branch(model, use_gelu=False)
    assert count > 0
    tagged = [m for m in model.modules() if getattr(m, "activation_name", "") == "squared_stochastic_schedule"]
    assert tagged
    assert all(m.act.__class__.__name__ == "SquaredReLU" for m in tagged)
    set_stochastic_squared_branch(model, use_gelu=True)
    assert all(m.act.__class__.__name__ == "SquaredGELU" for m in tagged)
    assert not any(isinstance(m.act, StochasticScheduledSquaredActivation) for m in tagged)
