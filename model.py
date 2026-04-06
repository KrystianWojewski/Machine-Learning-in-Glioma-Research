"""
Model do przewidywania Kp/Kpuu na podstawie struktury cząsteczki.
Jest to uproszczona wersja GraphDenseNet z MGraphDTA, ale bez części białkowej.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.nn import BatchNorm1d


class GraphConvBn(nn.Module):
    """
    Warstwa: GraphConv + BatchNorm + ReLU
    To jest podstawowy budulec naszej sieci.

    Co robi:
    1. GraphConv - wymienia informacje między sąsiednimi atomami
    2. BatchNorm - normalizuje wartości (żeby nie były za duże lub za małe)
    3. ReLU - dodaje nieliniowość (pozwala uczyć się skomplikowanych wzorców)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.norm = BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        # x: cechy atomów [liczba_atomów, in_channels]
        # edge_index: które atomy są połączone [2, liczba_wiązań]

        x = self.conv(x, edge_index)  # wymiana informacji między atomami
        x = self.norm(x)               # normalizacja
        x = F.relu(x)                  # nieliniowość
        return x


class DenseLayer(nn.Module):
    """
    Gęsta warstwa w bloku DenseNet.

    Co to znaczy "gęsta"? 
    W zwykłej sieci każda warstwa łączy się tylko z następną.
    W gęstej sieci KAŻDA warstwa łączy się ze WSZYSTKIMI poprzednimi.

    Dlaczego to robimy? 
    - Lepszy przepływ informacji
    - Łatwiejsze uczenie się (gradient nie znika)
    - Mniej parametrów (bo warstwy są węższe)

    Analogia: W zwykłej klasie uczeń rozmawia tylko z kolegą z ławki.
    W gęstej klasie KAŻDY rozmawia z KAŻDYM.
    """

    def __init__(self, in_channels, growth_rate=32, bn_size=4):
        super().__init__()
        # Najpierw rozszerzamy liczbę cech (bn_size * growth_rate)
        # Potem zmniejszamy do growth_rate
        self.conv1 = GraphConvBn(in_channels, bn_size * growth_rate)
        self.conv2 = GraphConvBn(bn_size * growth_rate, growth_rate)

    def forward(self, x, edge_index):
        # Zapamiętujemy wejście (dla połączeń gęstych)
        identity = x

        # Przetwarzamy
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        # Łączymy z wejściem (concat, nie add!)
        # To jest kluczowe dla DenseNet
        x = torch.cat([identity, x], dim=1)

        return x


class DenseBlock(nn.Module):
    """
    Blok złożony z kilku gęstych warstw.

    Jak działa:
    Warstwa 1: wejście [C] -> wyjście [C + growth_rate]
    Warstwa 2: wejście [C + growth_rate] -> wyjście [C + 2*growth_rate]
    Warstwa 3: wejście [C + 2*growth_rate] -> wyjście [C + 3*growth_rate]
    ...

    Liczba cech rośnie z każdą warstwą!
    """

    def __init__(self, num_layers, in_channels, growth_rate=32, bn_size=4):
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_layers):
            layer = DenseLayer(current_channels, growth_rate, bn_size)
            self.layers.append(layer)
            current_channels += growth_rate  # po każdej warstwie rośnie liczba cech

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class TransitionLayer(nn.Module):
    """
    Warstwa przejściowa między blokami.

    Po gęstym bloku liczba cech jest DUŻA (np. 300+).
    Przed kolejnym blokiem zmniejszamy ją o połowę.

    Dlaczego? Bo inaczej model miałby za dużo parametrów i byłby za wolny.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Zmniejszamy liczbę cech o połowę
        out_channels = in_channels // 2
        self.conv = GraphConvBn(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GraphDenseNet(nn.Module):
    """
    Główny model do kodowania cząsteczek.

    Struktura:
    ┌─────────────────────────────────────────────────────────────┐
    │  Wejście: cechy atomów [N_atomów, 74]                      │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Warstwa wstępna: GraphConvBn(74 → 32)                     │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Blok DenseBlock 1 (4 warstwy)                             │
    │  32 → 32 + 4*32 = 160 cech                                 │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  TransitionLayer: 160 → 80                                 │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Blok DenseBlock 2 (4 warstwy)                             │
    │  80 → 80 + 4*32 = 208 cech                                 │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  TransitionLayer: 208 → 104                                │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Blok DenseBlock 3 (4 warstwy)                             │
    │  104 → 104 + 4*32 = 232 cech                               │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Global Mean Pool: uśredniamy po wszystkich atomach        │
    │  [N_atomów, 232] → [1, 232]                                │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │  Wyjście: embedding cząsteczki [1, 232]                    │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, in_channels=74, growth_rate=32, num_layers_per_block=4):
        super().__init__()

        # Warstwa wstępna
        self.initial = GraphConvBn(in_channels, 32)

        # Blok 1
        self.block1 = DenseBlock(num_layers_per_block, 32, growth_rate)
        # Po bloku 1: 32 + 4*32 = 160 cech
        self.trans1 = TransitionLayer(160)  # 160 → 80

        # Blok 2
        self.block2 = DenseBlock(num_layers_per_block, 80, growth_rate)
        # Po bloku 2: 80 + 4*32 = 208 cech
        self.trans2 = TransitionLayer(208)  # 208 → 104

        # Blok 3
        self.block3 = DenseBlock(num_layers_per_block, 104, growth_rate)
        # Po bloku 3: 104 + 4*32 = 232 cech

        # Ostatnia warstwa liniowa (opcjonalna, można pominąć)
        self.final_linear = nn.Linear(232, 232)

    def forward(self, x, edge_index, batch):
        """
        x: [N_atomów, 74] - cechy wszystkich atomów
        edge_index: [2, E] - połączenia między atomami
        batch: [N_atomów] - dla każdego atomu, do której cząsteczki należy
        """
        # Warstwa wstępna
        x = self.initial(x, edge_index)

        # Blok 1
        x = self.block1(x, edge_index)
        x = self.trans1(x, edge_index)

        # Blok 2
        x = self.block2(x, edge_index)
        x = self.trans2(x, edge_index)

        # Blok 3
        x = self.block3(x, edge_index)

        # Globalne uśrednienie (pooling) – zbieramy informacje z wszystkich atomów
        x = global_mean_pool(x, batch)  # [batch_size, 232]

        # Ostatnia warstwa liniowa
        x = self.final_linear(x)

        return x


class MoleculePropertyPredictor(nn.Module):
    """
    Cały model: GraphDenseNet + głowa regresyjna.

    Zadanie: na podstawie grafu cząsteczki przewidzieć Kpuu (lub Kp).

    Co to jest "głowa regresyjna"?
    To kilka warstw, które zamieniają embedding cząsteczki (232 liczby)
    na JEDNĄ liczbę – przewidywaną wartość Kpuu.

    Dlaczego potrzebujemy głowy regresyjnej?
    Bo GraphDenseNet daje nam "zrozumienie" cząsteczki (embedding),
    ale nie umie jeszcze przewidywać konkretnej wartości.
    Głowa regresyjna uczy się, jak to zrozumienie zamienić na liczbę.
    """

    def __init__(self, in_channels=74, growth_rate=32, num_layers_per_block=4, dropout=0.2):
        super().__init__()

        # Encoder cząsteczek
        self.encoder = GraphDenseNet(
            in_channels, growth_rate, num_layers_per_block)

        # Głowa regresyjna (MLP)
        # 232 → 128 → 64 → 1
        self.regressor = nn.Sequential(
            nn.Linear(232, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, data):
        """
        data: obiekt z polami:
            - x: cechy atomów
            - edge_index: połączenia
            - edge_attr: cechy wiązań (nie używamy, ale trzymamy dla kompatybilności)
            - batch: przynależność do cząsteczek
        """
        # Kodujemy cząsteczkę
        embedding = self.encoder(data.x, data.edge_index, data.batch)

        # Przewidujemy wartość
        prediction = self.regressor(embedding)

        return prediction.view(-1)  # [batch_size]
