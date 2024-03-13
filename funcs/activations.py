import torch
import torch.nn as nn


class Hardsigm(nn.Module):
    def __init__(self,  low: float = 0, high: float = 1):
        super(Hardsigm, self).__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        return torch.clamp(x, self.low, self.high)

    def manual_grad(self, x):
        return (x >= self.low) & (x <= self.high)


class TanhSelf(nn.Module):
    @staticmethod
    def forward(x):
        return 0.5 + 0.5 * torch.tanh(x)

    @staticmethod
    def manual_grad(x):
        return (1 - torch.tanh(x)**2)*0.5


func_dict = {
    'sigmoid': nn.Sigmoid,
    'softmax': nn.Softmax,
    'tanh': TanhSelf,
    'relu': nn.ReLU,
    'hardsigm': Hardsigm,
    'x': nn.Identity,
}
