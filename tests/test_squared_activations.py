import torch

from models.atoms.activations import ScheduledSquaredActivation, build_activation


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
