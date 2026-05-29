import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors


def calculate_descriptors(smiles):
    """
    Oblicza zestaw deskryptorów fizykochemicznych dla cząsteczki.

    Returns:
        numpy array z deskryptorami lub None jeśli błąd
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = []

    # ========================================================================
    # 1. Podstawowe deskryptory molekularne (dla BBB penetration kluczowe!)
    # ========================================================================

    # logP (lipofilność) – kluczowy dla BBB
    logp = Descriptors.MolLogP(mol)
    descriptors.append(logp)

    # TPSA (Topological Polar Surface Area) – im niższa, tym lepsze przenikanie
    tpsa = Descriptors.TPSA(mol)
    descriptors.append(tpsa)

    # Masa cząsteczkowa (MW)
    mw = Descriptors.MolWt(mol)
    descriptors.append(mw)

    # Liczba akceptorów wiązań wodorowych (HBA)
    hba = Lipinski.NumHAcceptors(mol)
    descriptors.append(hba)

    # Liczba donorów wiązań wodorowych (HBD)
    hbd = Lipinski.NumHDonors(mol)
    descriptors.append(hbd)

    # ========================================================================
    # 2. Dodatkowe deskryptory (często używane w modelach BBB)
    # ========================================================================

    # Liczba rotowalnych wiązań
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    descriptors.append(rotatable_bonds)

    # Liczba pierścieni aromatycznych (poprawna nazwa: CalcNumAromaticRings)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    descriptors.append(aromatic_rings)

    # Liczba pierścieni (ogólnie)
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    descriptors.append(num_rings)

    # Liczba atomów
    num_atoms = mol.GetNumAtoms()
    descriptors.append(num_atoms)

    # Liczba atomów ciężkich (bez H)
    heavy_atoms = mol.GetNumHeavyAtoms()
    descriptors.append(heavy_atoms)

    # Stopień nasycenia (liczba wiązań podwójnych i potrójnych)
    num_double_bonds = 0
    num_triple_bonds = 0
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.rdchem.BondType.DOUBLE:
            num_double_bonds += 1
        elif bt == Chem.rdchem.BondType.TRIPLE:
            num_triple_bonds += 1
    descriptors.append(num_double_bonds)
    descriptors.append(num_triple_bonds)

    # ========================================================================
    # 3. Wskaźniki pochodne
    # ========================================================================

    # Stosunek HBA do HBD
    if hbd > 0:
        hba_hbd_ratio = hba / hbd
    else:
        hba_hbd_ratio = hba if hba > 0 else 0
    descriptors.append(hba_hbd_ratio)

    # Gęstość powierzchni polarnych (TPSA / MW)
    if mw > 0:
        polar_density = tpsa / mw
    else:
        polar_density = 0
    descriptors.append(polar_density)

    # ========================================================================
    # 4. Reguły lipofilności (dla BBB)
    # ========================================================================

    # Reguła "4/15" dla BBB: logP między 4 a 15? (uproszczona)
    # logP > 0 i logP < 15 (zazwyczaj)
    logp_good = 1 if 0 < logp < 15 else 0
    descriptors.append(logp_good)

    # TPSA < 90 Å² (dobrze dla BBB)
    tpsa_good = 1 if tpsa < 90 else 0
    descriptors.append(tpsa_good)

    # HBD < 3 (dobrze dla BBB)
    hbd_good = 1 if hbd < 3 else 0
    descriptors.append(hbd_good)

    # HBA < 7 (dobrze dla BBB)
    hba_good = 1 if hba < 7 else 0
    descriptors.append(hba_good)

    # ========================================================================
    # 5. Normalizacja
    # ========================================================================
    descriptors_array = np.array(descriptors, dtype=np.float32)

    # Normalizacja wybranych deskryptorów do zakresu [0, 1]
    # logP: typowy zakres -5 do 5, normalizacja do 0-1
    descriptors_array[0] = (descriptors_array[0] + 5) / 10.0
    descriptors_array[0] = max(0.0, min(1.0, descriptors_array[0]))  # clamp

    # TPSA: typowy zakres 0-200, normalizacja do 0-1
    descriptors_array[1] = min(descriptors_array[1] / 200.0, 1.0)

    # MW: typowy zakres 0-1000, normalizacja do 0-1
    descriptors_array[2] = min(descriptors_array[2] / 1000.0, 1.0)

    # Liczba pierścieni: normalizacja (max zwykle ~10)
    descriptors_array[7] = min(descriptors_array[7] / 10.0, 1.0)

    # Liczba atomów: normalizacja (max zwykle ~100)
    descriptors_array[8] = min(descriptors_array[8] / 100.0, 1.0)

    # Liczba atomów ciężkich: normalizacja
    descriptors_array[9] = min(descriptors_array[9] / 80.0, 1.0)

    return descriptors_array


def get_descriptor_names():
    """Zwraca nazwy deskryptorów w kolejności."""
    return [
        'logP_norm',
        'TPSA_norm',
        'MW_norm',
        'HBA',
        'HBD',
        'RotatableBonds',
        'AromaticRings',
        'NumRings_norm',
        'NumAtoms_norm',
        'HeavyAtoms_norm',
        'NumDoubleBonds',
        'NumTripleBonds',
        'HBA_HBD_ratio',
        'PolarDensity',
        'logP_good',
        'TPSA_good',
        'HBD_good',
        'HBA_good'
    ]


def test_descriptors():
    """Testuje obliczanie deskryptorów dla przykładowej cząsteczki."""
    smiles = "CC(C)(O)c1cc2nc(-c3cnc(N)nc3)nc(N3CCOCC3)c2s1"
    desc = calculate_descriptors(smiles)
    names = get_descriptor_names()

    print(f"\nDeskryptory dla cząsteczki:")
    for name, value in zip(names, desc):
        print(f"  {name}: {value:.4f}")

    return desc is not None


if __name__ == "__main__":
    test_descriptors()
