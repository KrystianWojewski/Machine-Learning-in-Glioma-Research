"""
Ładowanie danych i konwersja SMILES na grafy.
Dodano obsługę deskryptorów fizykochemicznych.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from graph_utils import smiles_to_graph
from descriptors import calculate_descriptors, get_descriptor_names


class PropertyDataset(Dataset):
    """
    Dataset dla przewidywania właściwości cząsteczek (Kp/Kpuu).
    Obsługuje zarówno grafy, jak i deskryptory fizykochemiczne.
    """

    def __init__(self, csv_file, target_col='Kpuu', use_descriptors=True, transform=None):
        """
        csv_file: ścieżka do pliku CSV
        target_col: nazwa kolumny z wartością do przewidzenia ('Kpuu' lub 'Kp')
        use_descriptors: czy dodawać deskryptory fizykochemiczne
        """
        self.df = pd.read_csv(csv_file)
        self.target_col = target_col
        self.use_descriptors = use_descriptors
        self.transform = transform

        # Usuń wiersze z brakującymi wartościami
        self.df = self.df.dropna(subset=[target_col])

        # Konwertuj SMILES na grafy i deskryptory
        print(f"Ładowanie {len(self.df)} cząsteczek...")
        self.graphs = []
        self.descriptors = []
        self.labels = []

        invalid_count = 0
        for idx, row in self.df.iterrows():
            smiles = row['SMILES']
            label = row[target_col]

            graph_dict = smiles_to_graph(smiles)

            if graph_dict is not None:
                # Graf
                data = Data(
                    x=graph_dict['x'],
                    edge_index=graph_dict['edge_index'],
                    edge_attr=graph_dict['edge_attr']
                )
                self.graphs.append(data)

                # Deskryptory
                if use_descriptors:
                    desc = calculate_descriptors(smiles)
                    if desc is not None:
                        self.descriptors.append(desc)
                    else:
                        # Jeśli deskryptory się nie udały, używamy zer
                        self.descriptors.append(
                            np.zeros(len(get_descriptor_names())))

                self.labels.append(label)
            else:
                invalid_count += 1

        print(f"Załadowano {len(self.graphs)} prawidłowych grafów")
        if invalid_count > 0:
            print(f"Odrzucono {invalid_count} nieprawidłowych SMILES")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.use_descriptors:
            desc = torch.tensor(self.descriptors[idx], dtype=torch.float32)
            return graph, desc, label
        else:
            return graph, label


def collate_fn(batch):
    """
    Łączy listę pojedynczych grafów w jeden batch.
    Obsługuje zarówno wersję z deskryptorami, jak i bez.
    """
    # Sprawdź czy batch zawiera deskryptory
    if len(batch[0]) == 3:
        graphs, descs, labels = zip(*batch)
        batch_graphs = Batch.from_data_list(graphs)
        batch_descs = torch.stack(descs)
        labels = torch.stack(labels)
        return batch_graphs, batch_descs, labels
    else:
        graphs, labels = zip(*batch)
        batch_graphs = Batch.from_data_list(graphs)
        labels = torch.stack(labels)
        return batch_graphs, labels


def create_dataloaders(csv_file, target_col='Kpuu', batch_size=32,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       use_descriptors=True, random_state=42):
    """
    Tworzy DataLoadery dla treningu, walidacji i testu.

    Args:
        use_descriptors: czy dodawać deskryptory fizykochemiczne
    """
    from sklearn.model_selection import train_test_split

    dataset = PropertyDataset(csv_file, target_col,
                              use_descriptors=use_descriptors)

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
    if use_descriptors:
        print(f"  Liczba deskryptorów: {len(get_descriptor_names())}")

    return train_loader, val_loader, test_loader
