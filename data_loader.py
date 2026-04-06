"""
Ładowanie danych i konwersja SMILES na grafy.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from graph_utils import smiles_to_graph


class PropertyDataset(Dataset):
    """
    Dataset dla przewidywania właściwości cząsteczek (Kp/Kpuu).

    Jak działa:
    1. Wczytuje plik CSV z kolumnami 'SMILES' i 'Kpuu' (lub 'Kp')
    2. Dla każdego SMILES tworzy graf (za pomocą smiles_to_graph)
    3. Przechowuje grafy i odpowiadające im wartości
    """

    def __init__(self, csv_file, target_col='Kpuu', transform=None):
        """
        csv_file: ścieżka do pliku CSV
        target_col: nazwa kolumny z wartością do przewidzenia ('Kpuu' lub 'Kp')
        """
        self.df = pd.read_csv(csv_file)
        self.target_col = target_col
        self.transform = transform

        # Usuń wiersze z brakującymi wartościami
        self.df = self.df.dropna(subset=[target_col])

        # Konwertuj SMILES na grafy
        print(f"Ładowanie {len(self.df)} cząsteczek...")
        self.graphs = []
        self.labels = []

        invalid_count = 0
        for idx, row in self.df.iterrows():
            smiles = row['SMILES']
            label = row[target_col]

            graph_dict = smiles_to_graph(smiles)

            if graph_dict is not None:
                # Konwertuj słownik na obiekt Data PyTorch Geometric
                data = Data(
                    x=graph_dict['x'],
                    edge_index=graph_dict['edge_index'],
                    edge_attr=graph_dict['edge_attr']
                )
                self.graphs.append(data)
                self.labels.append(label)
            else:
                invalid_count += 1

        print(f"Załadowano {len(self.graphs)} prawidłowych grafów")
        if invalid_count > 0:
            print(f"Odrzucono {invalid_count} nieprawidłowych SMILES")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_fn(batch):
    """
    Łączy listę pojedynczych grafów w jeden batch.

    Dlaczego to potrzebne?
    Każda cząsteczka ma inną liczbę atomów.
    PyTorch Geometric ma specjalną funkcję Batch.from_data_list,
    która łączy je w jeden duży graf (z odpowiednim polem 'batch').

    Pole 'batch' mówi, które atomy należą do której cząsteczki.
    Np. jeśli mamy 2 cząsteczki: pierwsza ma 5 atomów, druga ma 3 atomy,
    to batch = [0,0,0,0,0,1,1,1]
    """
    graphs, labels = zip(*batch)

    # Połącz grafy w jeden batch
    batch_graphs = Batch.from_data_list(graphs)

    # Połącz etykiety
    labels = torch.stack(labels)

    return batch_graphs, labels


def create_dataloaders(csv_file, target_col='Kpuu', batch_size=32,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       random_state=42):
    """
    Tworzy DataLoadery dla treningu, walidacji i testu.

    Podział danych:
    - 70% treningowe (train) - uczymy model na tych danych
    - 15% walidacyjne (val) - sprawdzamy, czy model się nie przeucza
    - 15% testowe (test) - oceniamy końcową wydajność

    Dlaczego potrzebujemy walidacji?
    Żeby wiedzieć, kiedy przestać trenować (early stopping).

    Dlaczego potrzebujemy testu?
    Żeby sprawdzić, czy model działa na całkowicie nowych danych.
    """
    from sklearn.model_selection import train_test_split

    dataset = PropertyDataset(csv_file, target_col)

    # Indeksy
    indices = list(range(len(dataset)))

    # Podział na train+val i test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=random_state
    )

    # Podział train+val na train i val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio_adjusted, random_state=random_state
    )

    # Tworzenie subsetów
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # DataLoadery
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"\nPodział danych:")
    print(f"  Treningowe: {len(train_dataset)}")
    print(f"  Walidacyjne: {len(val_dataset)}")
    print(f"  Testowe: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
