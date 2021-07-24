import torch

from digit_recognizer.models import get_model


def test_feedforward():
    x = torch.rand(1, 28, 28)
    model = get_model('feedforward')
    y = model(x)
    assert y.shape == (1, 10)


def test_convolutional():
    x = torch.rand(1, 28, 28)
    model = get_model('convolutional')
    y = model(x)
    assert y.shape == (1, 10)


def test_recurrent():
    x = torch.rand(1, 28, 28)
    model = get_model('recurrent')
    y = model(x)
    assert y.shape == (1, 10)
