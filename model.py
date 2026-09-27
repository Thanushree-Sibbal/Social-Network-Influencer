import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class InfluenceGCN(torch.nn.Module):
    """Two-layer GCN for a single scalar feature per node and two classes."""

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x.ndim != 2 or x.size(1) != 1:
            raise ValueError("Expected node features with shape [num_nodes, 1].")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("Expected edge_index with shape [2, num_edges].")
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
