import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Layer approximation using SiLU basis expansion.
    Replaces standard Linear layers in the Prediction Heads.
    """
    def __init__(self, in_features, out_features):
        super(KANLinear, self).__init__()
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.spline_weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Base linear path
        base = F.linear(x, self.base_weight)
        # Non-linear "spline" path (SiLU approximation)
        spline = F.linear(F.silu(x), self.spline_weight)
        return base + spline + self.bias