import torch
from torchvision import transforms as T
from torchvision.utils import make_grid


def tensor2pilimg(x):
    if type(x) == type([]):
        x = torch.cat(x, dim=2)
    return T.ToPILImage()(make_grid(x))
