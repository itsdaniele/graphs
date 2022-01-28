import torch.nn.functional as F


def get_activation_fn(activation):

    if activation == "relu":
        return F.relu
