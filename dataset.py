import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

dataset = Planetoid(
    root='data/Cora',
    name='Cora',
    transform=NormalizeFeatures()
)

data = dataset[0].to(device)

print("Dataset:", dataset)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Node feature dimension:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)
