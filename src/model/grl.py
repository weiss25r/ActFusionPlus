import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class GRLayer(nn.Module):
    def __init__(self):
        super(GRLayer, self).__init__()

    def forward(self, x, alpha):
        return GradientReversalFn.apply(x, alpha)
