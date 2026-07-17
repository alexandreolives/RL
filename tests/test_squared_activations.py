import torch

from models.atoms.activations import ScheduledSquaredActivation, build_activation
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
