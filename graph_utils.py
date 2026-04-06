import torch
import numpy as np
from rdkit import Chem

# Lista atomów (dla one-hot encoding)
ATOM_TYPES = [
    'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    'B', 'Si', 'Se', 'Te', 'As', 'Hg', 'Cd', 'Cu', 'Fe', 'Zn'
]

# Hybrydyzacje
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]


def one_hot_encoding(value, choices):
    """Tworzy wektor one-hot dla danej wartości."""
    return [1 if value == choice else 0 for choice in choices]


def get_atom_features(atom):
    """Generuje wektor cech dla atomu (74 wymiary)."""
    features = []

    # 1. Typ atomu (19)
    features.extend(one_hot_encoding(atom.GetSymbol(), ATOM_TYPES))

    # 2. Stopień (degree) - liczba sąsiadów (0-5, 6)
    degree = min(atom.GetDegree(), 5)
    degree_one_hot = [0] * 6
    degree_one_hot[degree] = 1
    features.extend(degree_one_hot)

    # 3. Formalny ładunek (-2 do 2)
    charge = atom.GetFormalCharge()
    charge_idx = min(max(charge + 2, 0), 4)
    charge_one_hot = [0] * 5
    charge_one_hot[charge_idx] = 1
    features.extend(charge_one_hot)

    # 4. Liczba atomów wodoru (0-4)
    num_h = min(atom.GetTotalNumHs(), 4)
    h_one_hot = [0] * 5
    h_one_hot[num_h] = 1
    features.extend(h_one_hot)

    # 5. Hybrydyzacja (5)
    hybrid = atom.GetHybridization()
    features.extend(one_hot_encoding(hybrid, HYBRIDIZATIONS))

    # 6. Czy aromatyczny (1)
    features.append(1 if atom.GetIsAromatic() else 0)

    # 7. Czy w pierścieniu (1)
    features.append(1 if atom.IsInRing() else 0)

    # Dopełnij do 74 (jeśli potrzeba)
    while len(features) < 74:
        features.append(0)

    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    """Generuje wektor cech dla wiązania (12 wymiarów)."""
    features = []

    # 1. Typ wiązania (4)
    bond_type = bond.GetBondType()
    bond_one_hot = [
        1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
        1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0,
    ]
    features.extend(bond_one_hot)

    # 2. Stereochemia (4)
    stereo = bond.GetStereo()
    stereo_one_hot = [
        1 if stereo == Chem.rdchem.BondStereo.STEREONONE else 0,
        1 if stereo == Chem.rdchem.BondStereo.STEREOANY else 0,
        1 if stereo == Chem.rdchem.BondStereo.STEREOZ else 0,
        1 if stereo == Chem.rdchem.BondStereo.STEREOE else 0,
    ]
    features.extend(stereo_one_hot)

    # 3. Czy wiązanie w pierścieniu (1)
    features.append(1 if bond.IsInRing() else 0)

    # Dopełnij do 12
    while len(features) < 12:
        features.append(0)

    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles):
    """Konwertuje SMILES na graf dla PyTorch Geometric."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Cechy atomów
    x = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # Krawędzie i ich cechy
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Dodaj w obie strony (graf nieskierowany)
        edge_index.append([i, j])
        edge_index.append([j, i])

        feat = get_bond_features(bond)
        edge_attr.append(feat)
        edge_attr.append(feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr
    }
