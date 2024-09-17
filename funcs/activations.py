import torch
import torch.nn as nn
import torch.nn.functional as F


class WTA(nn.Module):
    def __init__(self):
        super(WTA, self).__init__()

    def forward(self, x):
        # WTA among layers
        layer_mean = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=-1).reshape(x.shape[0], x.shape[1], 1, 1)
        x = x - layer_mean
        x = F.relu(x)
        # WTA among channels
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.relu(x)

        return x


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
    'wta':WTA,
}
