import torch
from ignite.metrics import FID, InceptionScore


def test_identical_features_have_zero_fid():
    x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    metric = FID(num_features=2)
    metric.update((x, x))
    assert abs(metric.compute()) < 1e-7


def test_constant_probabilities_have_is_one():
    metric = InceptionScore(num_features=3)
    metric.update(torch.tensor([[0.2, 0.3, 0.5]]).repeat(10, 1))
    assert abs(metric.compute() - 1.) < 1e-6
