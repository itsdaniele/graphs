from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .util import get_activation_fn


class GCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        depth,
        skip_connections=False,
        activation="relu",
    ):
        super().__init__()

        self.skip_connections = skip_connections
        self.hidden_dim = hidden_dim

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(depth):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.activation_fn = get_activation_fn(activation)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for conv in self.conv_layers:
            x = self.activation_fn(conv(x, edge_index, edge_weight)) + (
                0.0 if not self.skip_connections else x
            )

        return x
