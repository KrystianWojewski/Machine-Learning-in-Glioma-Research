"""
Wszystkie modele używane w projekcie:
1. GCN (Graph Convolutional Network) - baseline
2. GIN (Graph Isomorphism Network) - teoretycznie najsilniejsza
3. ResidualGCN - nasza główna architektura (uproszczona)
4. GraphDenseNet - nasza główna architektura (gęsta)
5. DescriptorOnly - tylko deskryptory fizykochemiczne
6. Hybrid - GNN + deskryptory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GraphConv, global_mean_pool
from torch.nn import BatchNorm1d


# ============================================================================
# 1. GCN Model (baseline)
# ============================================================================

class GCNModel(nn.Module):
    def __init__(self, in_channels=74, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.dropout = dropout

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.regressor(x)

        return x.view(-1)


# ============================================================================
# 2. GIN Model (Graph Isomorphism Network)
# ============================================================================

class GINModel(nn.Module):
    def __init__(self, in_channels=74, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(self.mlp1)
        self.bn1 = BatchNorm1d(hidden_dim)

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(self.mlp2)
        self.bn2 = BatchNorm1d(hidden_dim)

        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINConv(self.mlp3)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.dropout = dropout

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.regressor(x)

        return x.view(-1)


# ============================================================================
# 3. ResidualGCN (uproszczona, działająca wersja, Residual GCN)
# ============================================================================

class ResidualGCNModel(nn.Module):
    def __init__(self, in_channels=74, hidden_dim=128, dropout=0.2):
        super().__init__()

        # Warstwy konwolucyjne
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)

        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim)

        self.dropout = dropout

        # Regresor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + x2)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4 + x3)
        x4 = F.dropout(x4, p=self.dropout, training=self.training)

        x = global_mean_pool(x4, batch)

        x = self.regressor(x)

        return x.view(-1)

# ============================================================================
# 4. GraphDenseNet Model
# ============================================================================


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.norm = BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(in_channels, bn_size * growth_rate)
        self.conv2 = GraphConvBn(bn_size * growth_rate, growth_rate)

    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = torch.cat([identity, x], dim=1)
        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate=32, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_layers):
            layer = DenseLayer(current_channels, growth_rate, bn_size)
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.conv = GraphConvBn(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GraphDenseNetModel(nn.Module):
    def __init__(self, in_channels=74, growth_rate=32, num_layers_per_block=4, dropout=0.2):
        super().__init__()

        self.initial = GraphConvBn(in_channels, 32)

        self.block1 = DenseBlock(num_layers_per_block, 32, growth_rate)
        self.trans1 = TransitionLayer(160)

        self.block2 = DenseBlock(num_layers_per_block, 80, growth_rate)
        self.trans2 = TransitionLayer(208)

        self.block3 = DenseBlock(num_layers_per_block, 104, growth_rate)

        self.final_linear = nn.Linear(232, 232)

        self.dropout = dropout

        self.regressor = nn.Sequential(
            nn.Linear(232, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Warstwa wstępna: 74 → 32
        x = self.initial(x, edge_index)

        # DenseBlock 1: 32 → 160
        x = self.block1(x, edge_index)
        # Transition 1: 160 → 80
        x = self.trans1(x, edge_index)

        # DenseBlock 2: 80 → 208
        x = self.block2(x, edge_index)
        # Transition 2: 208 → 104
        x = self.trans2(x, edge_index)

        # DenseBlock 3: 104 → 232
        x = self.block3(x, edge_index)

        # Global Mean Pool: [N_atomów, 232] → [1, 232]
        x = global_mean_pool(x, batch)

        # Ostatnia warstwa liniowa
        x = self.final_linear(x)

        # Regresja
        x = self.regressor(x)

        return x.view(-1)


# ============================================================================
# 5. DescriptorOnly Model
# ============================================================================


class DescriptorOnlyModel(nn.Module):
    def __init__(self, descriptor_dim=18, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, descriptors):
        x = self.encoder(descriptors)
        x = self.regressor(x)
        return x.view(-1)


# ============================================================================
# 6. Hybrid Model (GNN + Descriptors)
# ============================================================================

class HybridModel(nn.Module):
    def __init__(self, descriptor_dim=18, hidden_dim=128, dropout=0.2):
        super().__init__()

        self.gnn_conv1 = GCNConv(74, hidden_dim)
        self.gnn_bn1 = BatchNorm1d(hidden_dim)
        self.gnn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gnn_bn2 = BatchNorm1d(hidden_dim)
        self.gnn_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.gnn_bn3 = BatchNorm1d(hidden_dim)

        self.desc_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        combined_dim = hidden_dim + (hidden_dim // 2)
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.dropout = dropout

    def forward(self, data, descriptors):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gnn_conv1(x, edge_index)
        x = self.gnn_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gnn_conv2(x, edge_index)
        x = self.gnn_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gnn_conv3(x, edge_index)
        x = self.gnn_bn3(x)
        x = F.relu(x)

        graph_embedding = global_mean_pool(
            x, batch)

        desc_embedding = self.desc_encoder(
            descriptors)

        combined = torch.cat([graph_embedding, desc_embedding], dim=1)

        output = self.regressor(combined)

        return output.view(-1)


def get_model(model_type, descriptor_dim=None, **kwargs):
    models_map = {
        'GCN': GCNModel,
        'GIN': GINModel,
        'ResidualGCN': ResidualGCNModel,
        'GraphDenseNet': GraphDenseNetModel,
        'DescriptorOnly': DescriptorOnlyModel,
        'Hybrid': HybridModel
    }

    if model_type not in models_map:
        raise ValueError(
            f"Nieznany typ modelu: {model_type}. Dostępne: {list(models_map.keys())}")

    if model_type in ['DescriptorOnly', 'Hybrid']:
        if descriptor_dim is None:
            from descriptors import get_descriptor_names
            descriptor_dim = len(get_descriptor_names())
        return models_map[model_type](descriptor_dim=descriptor_dim, **kwargs)
    else:
        return models_map[model_type](**kwargs)


def get_all_model_names():
    return ['GCN', 'GIN', 'ResidualGCN', 'GraphDenseNet', 'DescriptorOnly', 'Hybrid']
