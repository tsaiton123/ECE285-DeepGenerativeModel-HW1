import torch
from chest_vae.model import VAE, loss_terms
from chest_vae.data import patient_key


def test_loss_reduction_and_standard_normal_kl():
    x = torch.zeros(2, 1, 64, 64)
    mu = torch.zeros(2, 8)
    loss, bce, kl = loss_terms(x, x, mu, mu)
    assert torch.allclose(bce, 4096 * torch.log(torch.tensor(2.)))
    assert kl.item() == 0
    assert loss == bce


def test_sampling_shape_and_backward():
    torch.manual_seed(3)
    model = VAE(8, 8)
    x = torch.rand(2, 1, 64, 64)
    logits, mu, logvar = model(x)
    assert logits.shape == x.shape
    assert mu.shape == logvar.shape == (2, 8)
    loss, _, _ = loss_terms(logits, x, mu, logvar)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.mu.weight.grad.abs().sum() > 0
    assert model.logvar.weight.grad.abs().sum() > 0
    assert model.decode(torch.randn(3, 8)).shape == (3, 1, 64, 64)


def test_patient_grouping():
    assert patient_key("person123_bacteria_4.jpeg") == patient_key("person123_virus_5.jpeg")
    assert patient_key("NORMAL2-IM-1427-0001.jpeg") == "NORMAL2-IM-1427"
    assert patient_key("IM-0115-0001.jpeg") == "IM-0115"
