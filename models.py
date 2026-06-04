"""
Wszystkie modele używane w projekcie:
1. GCN (Graph Convolutional Network) - baseline
2. GIN (Graph Isomorphism Network) - teoretycznie najsilniejsza
3. GraphDenseNet - nasza główna architektura (uproszczona)
4. DescriptorOnly - tylko deskryptory fizykochemiczne
5. Hybrid - GNN + deskryptory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch.nn import BatchNorm1d


# ============================================================================
# 1. GCN Model (baseline)
# ============================================================================

class GCNModel(nn.Module):
    """
    Graph Convolutional Network - prosta, klasyczna architektura.
    """

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
    """
    Graph Isomorphism Network - teoretycznie najsilniejsza architektura GNN.
    """

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
# 3. GraphDenseNet (uproszczona, działająca wersja, Residual GCN)
# ============================================================================

class GraphDenseNetModel(nn.Module):
    """
    Uproszczona wersja Graph Dense Network.
    Używa warstw GraphConv z połączeniami gęstymi (skip connections).
    """

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

        # Warstwa 1
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Warstwa 2 (skip connection)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)  # Dodajemy wejście (skip connection)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Warstwa 3 (skip connection)
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3 + x2)  # Dodajemy wejście
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # Warstwa 4 (skip connection)
        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4 + x3)  # Dodajemy wejście
        x4 = F.dropout(x4, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x4, batch)

        # Regresja
        x = self.regressor(x)

        return x.view(-1)


# ============================================================================
# 4. DescriptorOnly Model
# ============================================================================

class DescriptorOnlyModel(nn.Module):
    """Model tylko deskryptorowy – do porównania z modelem grafowym."""

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
# 5. Hybrid Model (GNN + Descriptors)
# ============================================================================

class HybridModel(nn.Module):
    """Model hybrydowy: GNN (na grafie) + MLP (na deskryptorach)."""

    def __init__(self, descriptor_dim=18, hidden_dim=128, dropout=0.2):
        super().__init__()

        # GNN encoder
        self.gnn_conv1 = GCNConv(74, hidden_dim)
        self.gnn_bn1 = BatchNorm1d(hidden_dim)
        self.gnn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gnn_bn2 = BatchNorm1d(hidden_dim)
        self.gnn_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.gnn_bn3 = BatchNorm1d(hidden_dim)

        # Descriptor encoder
        self.desc_encoder = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        # Regresor (łączy embedding z GNN i deskryptorów)
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

        # GNN forward
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

        # Global pooling
        graph_embedding = global_mean_pool(
            x, batch)  # [batch_size, hidden_dim]

        # Descriptor forward
        desc_embedding = self.desc_encoder(
            descriptors)  # [batch_size, hidden_dim/2]

        # Konkatenacja
        combined = torch.cat([graph_embedding, desc_embedding], dim=1)

        # Regresja
        output = self.regressor(combined)

        return output.view(-1)


# ============================================================================
# 6. Użyteczności
# ============================================================================

def get_model(model_type, descriptor_dim=None, **kwargs):
    """Zwraca odpowiedni model na podstawie typu."""
    models_map = {
        'GCN': GCNModel,
        'GIN': GINModel,
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
    """Zwraca listę wszystkich dostępnych modeli."""
    return ['GCN', 'GIN', 'GraphDenseNet', 'DescriptorOnly', 'Hybrid']
