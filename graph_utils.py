import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

# ============================================================================
# DEFINITIONS FOR ATOM FEATURES (74 dimensions total)
# ============================================================================

ATOM_TYPES = [
    'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    'B', 'Si', 'Se', 'Te', 'As', 'Hg', 'Cd', 'Cu', 'Fe', 'Zn'
]

DEGREE_SIZE = 6

CHARGE_SIZE = 5

HYDROGEN_SIZE = 5

HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2
]

ELECTRONEGATIVITY = {
    'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'P': 2.19,
    'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'B': 2.04,
    'Si': 1.90, 'Se': 2.55, 'Te': 2.10, 'As': 2.18, 'Hg': 2.00,
    'Cd': 1.69, 'Cu': 1.90, 'Fe': 1.83, 'Zn': 1.65, 'H': 2.20,
    'default': 2.0
}

ATOMIC_MASS = {
    'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'P': 30.97,
    'S': 32.06, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90, 'B': 10.81,
    'Si': 28.09, 'Se': 78.96, 'Te': 127.60, 'As': 74.92, 'Hg': 200.59,
    'Cd': 112.41, 'Cu': 63.55, 'Fe': 55.85, 'Zn': 65.38, 'H': 1.01,
    'default': 12.0
}


def one_hot_encoding(value, choices):
    return [1 if value == choice else 0 for choice in choices]


def get_ring_info(mol, atom_idx):
    ring_info = mol.GetRingInfo()
    num_rings = ring_info.NumAtomRings(atom_idx)
    min_ring_size = ring_info.MinAtomRingSize(atom_idx)
    return num_rings, min_ring_size


def get_atom_features_full(mol, atom, atom_idx):
    """
    Args:
        mol: obiekt RDKit Mol (potrzebny do informacji o pierścieniach)
        atom: obiekt RDKit Atom
        atom_idx: indeks atomu w molekule
    """
    features = []
    atom_symbol = atom.GetSymbol()

    # 1. Typ atomu (19)
    features.extend(one_hot_encoding(atom_symbol, ATOM_TYPES))

    # 2. Stopień (degree) – liczba sąsiadów (0-5, 6+)
    degree = min(atom.GetDegree(), 5)
    degree_one_hot = [0] * DEGREE_SIZE
    degree_one_hot[degree] = 1
    features.extend(degree_one_hot)

    # 3. Formalny ładunek (-2 do 2)
    charge = atom.GetFormalCharge()
    charge_idx = min(max(charge + 2, 0), 4)
    charge_one_hot = [0] * CHARGE_SIZE
    charge_one_hot[charge_idx] = 1
    features.extend(charge_one_hot)

    # 4. Liczba atomów wodoru (0-4, 5+)
    num_h = min(atom.GetTotalNumHs(), 4)
    h_one_hot = [0] * HYDROGEN_SIZE
    h_one_hot[num_h] = 1
    features.extend(h_one_hot)

    # 5. Hybrydyzacja (5 typów)
    hybrid = atom.GetHybridization()
    hybrid_one_hot = [0] * len(HYBRIDIZATIONS)
    if hybrid in HYBRIDIZATIONS:
        idx = HYBRIDIZATIONS.index(hybrid)
        hybrid_one_hot[idx] = 1
    features.extend(hybrid_one_hot)

    # 6. Czy atom jest aromatyczny (1)
    features.append(1 if atom.GetIsAromatic() else 0)

    # 7. Czy atom jest w pierścieniu (1)
    features.append(1 if atom.IsInRing() else 0)

    # 8. Masa atomowa (normalizowana do zakresu [0, 1])
    atomic_mass = ATOMIC_MASS.get(atom_symbol, ATOMIC_MASS['default'])
    features.append(atomic_mass / 200.0)  # max ~200 dla Hg

    # 9. Elektroujemność (normalizowana do [0, 1])
    eneg = ELECTRONEGATIVITY.get(atom_symbol, ELECTRONEGATIVITY['default'])
    features.append(eneg / 5.0)  # max F ~4.0, zapas 5.0

    # 10. Liczba pierścieni, w których atom uczestniczy (normalizowana)
    num_rings, min_ring_size = get_ring_info(mol, atom_idx)
    features.append(min(num_rings, 5) / 5.0)

    # 11. Rozmiar najmniejszego pierścienia (normalizowany)
    if min_ring_size == 0:
        min_ring_size = 6
    features.append(min(6, min_ring_size) / 6.0)

    # 12. Walencja (normalizowana)
    valence = atom.GetValence(Chem.ValenceType.IMPLICIT) + \
        atom.GetValence(Chem.ValenceType.EXPLICIT)
    if valence < 0:
        valence = 0
    features.append(min(6, valence) / 6.0)

    # 13. Liczba wiązań podwójnych (normalizowana)
    double_bonds = 0
    for bond in atom.GetBonds():
        if bond.GetBondType() == rdchem.BondType.DOUBLE:
            double_bonds += 1
    features.append(min(3, double_bonds) / 3.0)

    # 14. Liczba wiązań potrójnych (normalizowana)
    triple_bonds = 0
    for bond in atom.GetBonds():
        if bond.GetBondType() == rdchem.BondType.TRIPLE:
            triple_bonds += 1
    features.append(min(2, triple_bonds) / 2.0)

    # 15. Czy atom jest donorem wiązania wodorowego? (1)
    is_h_donor = 0
    if atom.GetTotalNumHs() > 0 and atom_symbol in ['O', 'N', 'S']:
        is_h_donor = 1
    features.append(is_h_donor)

    # 16. Czy atom jest akceptorem wiązania wodorowego? (1)
    is_h_acceptor = 0
    if atom_symbol in ['O', 'N', 'F'] and atom.GetTotalNumHs() == 0:
        is_h_acceptor = 1
    features.append(is_h_acceptor)

    # 17. Stopień z uwzględnieniem wodoru (normalizowany)
    total_degree = atom.GetDegree() + atom.GetTotalNumHs()
    features.append(min(8, total_degree) / 8.0)

    # 18. Liczba sąsiadujących atomów tlenu (normalizowana)
    o_neighbors = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'O':
            o_neighbors += 1
    features.append(min(4, o_neighbors) / 4.0)

    # 19. Liczba sąsiadujących atomów azotu (normalizowana)
    n_neighbors = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'N':
            n_neighbors += 1
    features.append(min(4, n_neighbors) / 4.0)

    # 20. Liczba sąsiadujących atomów fluoru (normalizowana)
    f_neighbors = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'F':
            f_neighbors += 1
    features.append(min(4, f_neighbors) / 4.0)

    # 21. Liczba sąsiadujących atomów chloru (normalizowana)
    cl_neighbors = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'Cl':
            cl_neighbors += 1
    features.append(min(4, cl_neighbors) / 4.0)

    # 22. Liczba sąsiadujących atomów siarki (normalizowana)
    s_neighbors = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'S':
            s_neighbors += 1
    features.append(min(4, s_neighbors) / 4.0)

    # 23. Formalny ładunek (wartość ciągła) – dodatkowa cecha
    features.append(charge / 5.0)  # normalizacja do [-0.4, 0.4]

    # 24-74. Dopełnienie do 74 zerami (dla kompatybilności)
    while len(features) < 74:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    features = []

    # 1. Typ wiązania (4)
    bond_type = bond.GetBondType()
    bond_one_hot = [
        1 if bond_type == rdchem.BondType.SINGLE else 0,
        1 if bond_type == rdchem.BondType.DOUBLE else 0,
        1 if bond_type == rdchem.BondType.TRIPLE else 0,
        1 if bond_type == rdchem.BondType.AROMATIC else 0,
    ]
    features.extend(bond_one_hot)

    # 2. Stereochemia (4)
    stereo = bond.GetStereo()
    stereo_one_hot = [
        1 if stereo == rdchem.BondStereo.STEREONONE else 0,
        1 if stereo == rdchem.BondStereo.STEREOANY else 0,
        1 if stereo == rdchem.BondStereo.STEREOZ else 0,
        1 if stereo == rdchem.BondStereo.STEREOE else 0,
    ]
    features.extend(stereo_one_hot)

    # 3. Czy wiązanie w pierścieniu (1)
    features.append(1 if bond.IsInRing() else 0)

    # 4. Długość wiązania (normalizowana)
    bond_length = 1.5
    if bond_type == rdchem.BondType.SINGLE:
        bond_length = 1.54
    elif bond_type == rdchem.BondType.DOUBLE:
        bond_length = 1.34
    elif bond_type == rdchem.BondType.TRIPLE:
        bond_length = 1.20
    elif bond_type == rdchem.BondType.AROMATIC:
        bond_length = 1.40
    features.append(bond_length / 2.0)

    # 5. Czy wiązanie jest skoniugowane (1)
    features.append(1 if bond.GetIsConjugated() else 0)

    # Dopełnij do 12
    while len(features) < 12:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = []
    for i, atom in enumerate(mol.GetAtoms()):
        features = get_atom_features_full(mol, atom, i)
        x.append(features)
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

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
