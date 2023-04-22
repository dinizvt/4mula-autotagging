import torch.nn.functional as F
from torch.nn import BCELoss

def bce_loss(output, target):
    return BCELoss()(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)
