"""
Wizualizacja grafów molekularnych otrzymanych z SMILES.
Pokazuje, jak cząsteczka jest reprezentowana jako graf (atomy = węzły, wiązania = krawędzie).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.utils import to_networkx

from graph_utils import smiles_to_graph


def visualize_molecular_graph(smiles, save_path=None, figsize=(12, 10)):
    """
    Tworzy dwa widoki tej samej cząsteczki:
    1. Tradycyjny wzór chemiczny (RDKit)
    2. Graf molekularny (NetworkX) - taki, jaki widzi model

    Args:
        smiles: SMILES cząsteczki
        save_path: ścieżka do zapisu (opcjonalnie)
        figsize: rozmiar figury
    """

    # 1. Konwersja SMILES na graf
    graph_dict = smiles_to_graph(smiles)
    if graph_dict is None:
        print(f"Nie można przetworzyć SMILES: {smiles}")
        return

    # 2. Przygotowanie cząsteczki do wizualizacji RDKit
    mol = Chem.MolFromSmiles(smiles)

    # 3. Konwersja na NetworkX (do wizualizacji grafu)
    # Tworzymy prosty graf z cechami
    x = graph_dict['x'].numpy()
    edge_index = graph_dict['edge_index'].numpy()

    # Stwórz graf NetworkX
    G = nx.Graph()

    # Dodaj węzły (atomy) z cechami
    for i in range(len(x)):
        # Znajdź symbol atomu z RDKit
        if mol:
            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
        else:
            atom_symbol = f"Atom_{i}"
        G.add_node(i, label=atom_symbol, importance=0)  # importance na razie 0

    # Dodaj krawędzie (wiązania)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u < v:  # dodaj tylko raz (graf nieskierowany)
            G.add_edge(u, v)

    # 4. Rysowanie
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Lewy panel: tradycyjny wzór chemiczny (RDKit)
    img = Draw.MolToImage(mol, size=(400, 400))
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Tradycyjny wzór chemiczny\n(RDKit)',
                      fontsize=12, fontweight='bold')

    # Prawy panel: graf molekularny (NetworkX)
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Rysuj węzły (atomy)
    node_colors = ['lightblue'] * len(G.nodes())
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=node_colors,
                           node_size=800, alpha=0.9)

    # Rysuj krawędzie (wiązania)
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color='gray',
                           width=1.5, alpha=0.7)

    # Rysuj etykiety (symbole atomów)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, ax=axes[1], labels=labels,
                            font_size=10, font_weight='bold')

    axes[1].set_title('Graf molekularny\n(jak widzi go model GNN)',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Zapisano wizualizację: {save_path}")

    plt.show()

    # Dodatkowo: wypisz informacje o grafie
    print("\n" + "="*50)
    print("INFORMACJE O GRAFIE")
    print("="*50)
    print(f"Liczba atomów (węzłów): {len(x)}")
    print(
        f"Liczba wiązań (krawędzi): {edge_index.shape[1] // 2} (nieskierowanych)")
    print(f"Wymiar cech atomów: {x.shape[1]}")
    print(f"Wymiar cech wiązań: {graph_dict['edge_attr'].shape[1]}")

    return G, mol


def compare_multiple_molecules(smiles_list, names_list, save_path=None):
    """
    Porównuje wiele cząsteczek – pokazuje ich wzory chemiczne i grafy.

    Args:
        smiles_list: lista SMILES
        names_list: lista nazw (np. CHEMBL ID)
        save_path: ścieżka do zapisu
    """
    n = len(smiles_list)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4*n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (smiles, name) in enumerate(zip(smiles_list, names_list)):
        # Konwersja na graf
        graph_dict = smiles_to_graph(smiles)
        if graph_dict is None:
            print(f"Nie można przetworzyć: {name}")
            continue

        mol = Chem.MolFromSmiles(smiles)
        x = graph_dict['x'].numpy()
        edge_index = graph_dict['edge_index'].numpy()

        # Lewy panel: wzór chemiczny
        img = Draw.MolToImage(mol, size=(300, 300))
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'{name}\nWzór chemiczny', fontsize=10)

        # Prawy panel: graf
        G = nx.Graph()
        for j in range(len(x)):
            if mol:
                atom_symbol = mol.GetAtomWithIdx(j).GetSymbol()
            else:
                atom_symbol = f"A{j}"
            G.add_node(j, label=atom_symbol)

        for j in range(edge_index.shape[1]):
            u, v = edge_index[0, j], edge_index[1, j]
            if u < v:
                G.add_edge(u, v)

        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=30)

        nx.draw_networkx_nodes(G, pos, ax=axes[i, 1], node_color='lightblue',
                               node_size=400, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=axes[i, 1], edge_color='gray',
                               width=1, alpha=0.7)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, ax=axes[i, 1], labels=labels,
                                font_size=8)

        axes[i, 1].set_title(f'{name}\nGraf molekularny', fontsize=10)
        axes[i, 1].axis('off')

        # Dodaj informację o liczbie atomów
        axes[i, 0].text(0.5, -0.1, f'Atomy: {len(x)} | Wiązania: {edge_index.shape[1]//2}',
                        transform=axes[i, 0].transAxes, ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Zapisano porównanie: {save_path}")

    plt.show()


def visualize_graph_with_importance(smiles, importance, save_path=None):
    """
    Wizualizuje graf molekularny z kolorami wskazującymi ważność atomów.

    Args:
        smiles: SMILES cząsteczki
        importance: lista ważności atomów (z integrated_gradients)
        save_path: ścieżka do zapisu
    """
    graph_dict = smiles_to_graph(smiles)
    if graph_dict is None:
        print(f"Nie można przetworzyć SMILES: {smiles}")
        return

    mol = Chem.MolFromSmiles(smiles)
    x = graph_dict['x'].numpy()
    edge_index = graph_dict['edge_index'].numpy()

    # Stwórz graf NetworkX
    G = nx.Graph()
    for i in range(len(x)):
        if mol:
            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
        else:
            atom_symbol = f"A{i}"
        G.add_node(
            i, label=f"{atom_symbol}\n{importance[i]:.3f}", importance=importance[i])

    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u < v:
            G.add_edge(u, v)

    # Przygotuj kolory: czerwony (ważny) → niebieski (nieważny)
    importance_norm = (importance - importance.min()) / \
        (importance.max() - importance.min() + 1e-8)
    # RdYlGn_r: zielony=nieważny, czerwony=ważny
    node_colors = plt.cm.RdYlGn_r(importance_norm)

    # Rysowanie
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Lewy panel: wzór chemiczny z zaznaczonymi ważnymi atomami
    top_indices = np.argsort(importance)[-5:][::-1]
    highlight_atoms = [int(i) for i in top_indices]
    img = Draw.MolToImage(mol, size=(400, 400),
                          highlightAtoms=highlight_atoms,
                          highlightColor=(0.8, 0.2, 0.2))
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(
        'Wzór chemiczny\n(czerwone = najważniejsze atomy)', fontsize=12)

    # Prawy panel: graf z kolorami
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=node_colors,
                           node_size=600, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color='gray',
                           width=1.5, alpha=0.7)

    # Etykiety z symbolem atomu i ważnością
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=axes[1], labels=labels, font_size=9)

    axes[1].set_title('Graf molekularny\n(kolor = ważność atomu)', fontsize=12)
    axes[1].axis('off')

    # Dodaj colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                               norm=plt.Normalize(vmin=importance.min(), vmax=importance.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], shrink=0.7)
    cbar.set_label('Ważność atomu (Integrated Gradients)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Zapisano wizualizację ważności: {save_path}")

    plt.show()


if __name__ == "__main__":
    import os

    # Utwórz katalog na wizualizacje
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)

    # Przykładowe SMILES – weźmy różne typy cząsteczek
    examples = [
        # Prosta cząsteczka
        ("CC(C)(O)c1cc2nc(-c3cnc(N)nc3)nc(N3CCOCC3)c2s1",
         "Przykład 1: Związek heterocykliczny"),
        # Cząsteczka z pierścieniami aromatycznymi
        ("COc1ccc2c(c1)[C@@]13CCCC[C@@H]1[C@@H](C2)N(C)CC3",
         "Przykład 2: Układ pierścieni skondensowanych"),
        # Cząsteczka z fluorami
        ("O[C@]1(C(F)(F)F)CCCC[C@H]1Nc1ccc(F)cc1",
         "Przykład 3: Związek z atomami fluoru"),
    ]

    print("="*60)
    print("WIZUALIZACJA GRAFÓW MOLEKULARNYCH")
    print("="*60)

    # Wizualizacja pojedynczej cząsteczki
    smiles, name = examples[0]
    print(f"\n1. Wizualizacja: {name}")
    G, mol = visualize_molecular_graph(
        smiles, save_path=os.path.join(viz_dir, 'single_molecule_graph.png'))

    # Porównanie wielu cząsteczek
    print(f"\n2. Porównanie cząsteczek")
    smiles_list = [ex[0] for ex in examples]
    names_list = [ex[1] for ex in examples]
    compare_multiple_molecules(smiles_list, names_list,
                               save_path=os.path.join(viz_dir, 'multiple_molecules_comparison.png'))

    print("\n Wszystkie wizualizacje zostały zapisane w katalogu 'visualizations'")
