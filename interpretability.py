"""
Interpretowalność modelu – identyfikacja ważnych atomów i fragmentów cząsteczek.
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from torch_geometric.data import Data


def integrated_gradients(model, data, target_idx=0, steps=50):
    """
    Oblicza zintegrowane gradienty dla każdego atomu.

    Integrated Gradients to metoda atrybucji, która:
    1. Startuje z punktu bazowego (zero)
    2. Liniowo interpoluje między bazą a wejściem
    3. Całkuje gradienty po tej ścieżce

    Wynik: ważność każdego atomu dla predykcji
    """
    model.eval()

    # Upewnij się, że dane są na tym samym urządzeniu co model
    device = next(model.parameters()).device
    data = data.to(device)

    # Punkt bazowy (zerowe cechy atomów) – na tym samym urządzeniu
    baseline = torch.zeros_like(data.x)

    # Przygotuj interpolacje
    alphas = torch.linspace(0, 1, steps, device=device)
    interpolated = baseline + alphas.view(-1, 1, 1) * (data.x - baseline)

    # Oblicz gradienty dla każdej interpolacji
    grads = []
    for alpha_interp in interpolated:
        data_interp = Data(
            x=alpha_interp,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        ).to(device)
        data_interp.x.requires_grad_(True)

        output = model(data_interp)
        output[target_idx].backward()

        grad = data_interp.x.grad.clone()
        grads.append(grad)
        data_interp.x.grad.zero_()

    # Średnia ważona gradientów
    grads = torch.stack(grads)
    avg_grads = torch.trapz(grads, alphas, dim=0)

    # Atrybucja = (x - baseline) * średni gradient
    attributions = (data.x - baseline) * avg_grads
    # Suma po cechach dla każdego atomu
    node_importance = attributions.sum(dim=1).detach().cpu().numpy()

    return node_importance


def get_atom_contributions(smiles, model, device):
    """
    Dla pojedynczej cząsteczki oblicza ważność każdego atomu.
    """
    from graph_utils import smiles_to_graph

    # Konwersja SMILES na graf
    graph_dict = smiles_to_graph(smiles)
    if graph_dict is None:
        return None, None

    # Upewnij się, że batch jest na tym samym urządzeniu co model
    data = Data(
        x=graph_dict['x'],
        edge_index=graph_dict['edge_index'],
        edge_attr=graph_dict['edge_attr'],
        batch=torch.zeros(graph_dict['x'].shape[0], dtype=torch.long)
    ).to(device)

    # Oblicz ważność atomów
    importance = integrated_gradients(model, data)

    # Przygotuj cząsteczkę do wizualizacji
    mol = Chem.MolFromSmiles(smiles)

    return mol, importance


def visualize_important_atoms(smiles, model, device, save_path=None):
    """
    Wizualizuje cząsteczkę z kolorami wskazującymi ważność atomów.
    """
    mol, importance = get_atom_contributions(smiles, model, device)

    if mol is None:
        print(f"Nie można przetworzyć SMILES: {smiles}")
        return

    # Normalizacja ważności do zakresu [0,1] dla kolorów
    importance_norm = (importance - importance.min()) / \
        (importance.max() - importance.min() + 1e-8)

    print(f"\nCząsteczka: {smiles[:80]}...")
    print("Najważniejsze atomy (indeks, ważność):")
    top_indices = np.argsort(importance)[-5:][::-1]
    for idx in top_indices:
        atom = mol.GetAtomWithIdx(int(idx))  # <-- KONWERSJA NA int
        print(f"  Atom {int(idx)} ({atom.GetSymbol()}): {importance[idx]:.4f}")

    if save_path:
        # Prosta wizualizacja z zaznaczonymi ważnymi atomami
        highlight_atoms = [int(i) for i in top_indices[:3]]
        img = Draw.MolToImage(mol, size=(400, 400),
                              highlightAtoms=highlight_atoms,
                              highlightColor=(0.8, 0.2, 0.2))
        img.save(save_path)
        print(f"Zapisano wizualizację: {save_path}")

    return mol, importance


def analyze_dataset(model, device, csv_file, n_samples=10):
    """
    Analizuje wybrane przykłady ze zbioru testowego.
    Z DODANYM CHEMBL ID.
    """
    from graph_utils import smiles_to_graph

    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['pKpuu'])

    # Wybierz próbki: 5 z wysokim pKpuu (dobrze przenikające) i 5 z niskim
    df_sorted = df.sort_values('pKpuu')
    low_samples = df_sorted.head(n_samples // 2)
    high_samples = df_sorted.tail(n_samples // 2)

    samples = pd.concat([low_samples, high_samples])

    print("\n" + "="*60)
    print("ANALIZA PRZYKŁADÓW - IDENTYFIKACJA WAŻNYCH ATOMÓW")
    print("="*60)

    results = []
    for idx, row in samples.iterrows():
        smiles = row['SMILES']
        true_value = row['pKpuu']
        # <-- POBIERZ CHEMBL ID
        chembl_id = row.get('CompoundID', f'CHEMBL_{idx}')

        # Konwersja SMILES na graf
        graph_dict = smiles_to_graph(smiles)
        if graph_dict is None:
            print(f"Pomijam (nieprawidłowy SMILES): {smiles[:50]}...")
            continue

        data = Data(
            x=graph_dict['x'],
            edge_index=graph_dict['edge_index'],
            edge_attr=graph_dict['edge_attr'],
            batch=torch.zeros(graph_dict['x'].shape[0], dtype=torch.long)
        ).to(device)

        # Przewidywanie
        model.eval()
        with torch.no_grad():
            pred = model(data).cpu().numpy()[0]

        # Ważność atomów
        importance = integrated_gradients(model, data)

        results.append({
            'chembl_id': chembl_id,  # <-- DODANE CHEMBL ID
            'smiles': smiles,
            'true': true_value,
            'pred': pred,
            'importance': importance,
            'num_atoms': len(importance)
        })

        print(
            f"\n--- {chembl_id} (pKpuu: true={true_value:.3f}, pred={pred:.3f}) ---")
        print(f"SMILES: {smiles[:80]}...")
        print(f"Liczba atomów: {len(importance)}")
        print(f"Średnia ważność: {importance.mean():.4f}")
        print(f"Max ważność: {importance.max():.4f}")

        # Pokaż 3 najważniejsze atomy
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            top_indices = np.argsort(importance)[-3:][::-1]
            print("Najważniejsze atomy (ważność):")
            for atom_idx in top_indices:
                atom_idx_int = int(atom_idx)
                atom = mol.GetAtomWithIdx(atom_idx_int)
                print(
                    f"  Atom {atom_idx_int} ({atom.GetSymbol()}): {importance[atom_idx]:.4f}")

    return results


def plot_atom_importance_for_molecules(results, save_path=None, top_k=5):
    """
    Rysuje wykres słupkowy pokazujący najważniejsze atomy dla każdej cząsteczki.

    Args:
        results: lista wyników z analyze_dataset
        save_path: ścieżka do zapisu
        top_k: liczba najważniejszych atomów do pokazania
    """
    # Wybierz tylko kilka cząsteczek do pokazania (żeby wykres był czytelny)
    # Weźmy 4 cząsteczki: 2 z najwyższym pKpuu (słabe przenikanie) i 2 z najniższym (dobre przenikanie)

    sorted_results = sorted(results, key=lambda x: x['true'])
    low_pkpuu = sorted_results[:2]   # dobre przenikanie (niskie pKpuu)
    high_pkpuu = sorted_results[-2:]  # słabe przenikanie (wysokie pKpuu)
    selected = low_pkpuu + high_pkpuu

    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for i, res in enumerate(selected):
        importance = res['importance']
        mol = Chem.MolFromSmiles(res['smiles'])

        # Pobierz symbole atomów dla najważniejszych atomów
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_atoms = []
        top_importance = []

        for idx in top_indices:
            atom = mol.GetAtomWithIdx(int(idx))
            top_atoms.append(f"{atom.GetSymbol()}{int(idx)}")
            top_importance.append(importance[idx])

        # Kolorowanie: zielony dla dodatniej ważności (zwiększa przenikanie),
        # czerwony dla ujemnej (zmniejsza przenikanie)
        colors = ['green' if val > 0 else 'red' for val in top_importance]

        # Wykres słupkowy
        bars = axes[i].barh(top_atoms, top_importance, color=colors, alpha=0.7)
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].set_xlabel('Ważność atomu (Integrated Gradients)', fontsize=10)

        # Tytuł z informacją o cząsteczce
        if res['true'] < 0:
            penetration = "DOBRE przenikanie"
        else:
            penetration = "SŁABE przenikanie"

        axes[i].set_title(f"{res['chembl_id']} | {penetration}\n"
                          f"pKpuu: true={res['true']:.2f}, pred={res['pred']:.2f} | "
                          f"błąd={abs(res['true']-res['pred']):.2f}", fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano wykres ważności atomów: {save_path}")
    plt.show()


def plot_all_atoms_importance(results, save_path=None):
    """
    Rysuje wykres punktowy wszystkich atomów we wszystkich cząsteczkach,
    z kolorami wskazującymi typ cząsteczki (dobre vs słabe przenikanie).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Przygotuj dane
    all_importances = []
    all_types = []  # 'good' dla niskiego pKpuu, 'bad' dla wysokiego

    for i, res in enumerate(results):
        importance = res['importance']
        # niskie pKpuu = dobre przenikanie
        is_good_penetration = res['true'] < 0

        for imp in importance:
            all_importances.append(imp)
            all_types.append(
                'Dobre przenikanie' if is_good_penetration else 'Słabe przenikanie')

    # Stwórz DataFrame do łatwiejszego rysowania
    df_plot = pd.DataFrame({
        'importance': all_importances,
        'type': all_types
    })

    # Wykres punktowy (strip plot)
    # Dla każdej grupy, każdy punkt to jeden atom
    positions = [0, 1]
    # niebieski dla dobrych, pomarańczowy dla słabych
    colors = ['#2E86AB', '#F18F01']

    # Użyjemy violin plot + strip plot dla lepszej wizualizacji
    parts = ax.violinplot([df_plot[df_plot['type'] == 'Dobre przenikanie']['importance'],
                           df_plot[df_plot['type'] == 'Słabe przenikanie']['importance']],
                          positions=positions, showmeans=False, showmedians=True)

    # Ustaw kolory dla violin plot
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.5)

    # Dodaj punkty (strip plot)
    for i, (type_name, color) in enumerate(zip(['Dobre przenikanie', 'Słabe przenikanie'], colors)):
        data = df_plot[df_plot['type'] == type_name]['importance']
        # Dodaj trochę szumu, żeby punkty się nie nakładały
        x_jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.3, s=10, color=color)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Próg (0)')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Dobre przenikanie\n(niskie pKpuu)',
                       'Słabe przenikanie\n(wysokie pKpuu)'])
    ax.set_ylabel('Ważność atomu (Integrated Gradients)', fontsize=12)
    ax.set_title(
        'Porównanie ważności atomów\nw cząsteczkach dobrze i słabo przenikających przez BBB', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Dodaj statystyki
    good_median = df_plot[df_plot['type'] ==
                          'Dobre przenikanie']['importance'].median()
    bad_median = df_plot[df_plot['type'] ==
                         'Słabe przenikanie']['importance'].median()
    ax.text(0, -0.15, f'Mediana: {good_median:.4f}',
            ha='center', fontsize=9, color=colors[0])
    ax.text(1, -0.15, f'Mediana: {bad_median:.4f}',
            ha='center', fontsize=9, color=colors[1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano wykres porównawczy: {save_path}")
    plt.show()


def plot_atom_importance_heatmap(results, save_path=None):
    """
    Rysuje heatmapę ważności atomów dla wszystkich cząsteczek.
    Każdy wiersz = jedna cząsteczka, każda kolumna = atom (posortowany według ważności).
    """
    # Znajdź maksymalną liczbę atomów
    max_atoms = max([res['num_atoms'] for res in results])

    # Stwórz macierz ważności (padding z NaN)
    importance_matrix = []
    labels = []

    for res in results:
        importance = res['importance']
        # Posortuj malejąco według ważności
        sorted_imp = np.sort(importance)[::-1]
        # Dopełnij NaN do max_atoms
        padded = np.pad(sorted_imp, (0, max_atoms - len(sorted_imp)),
                        constant_values=np.nan)
        importance_matrix.append(padded)
        labels.append(f"{res['chembl_id']}\n(true={res['true']:.1f})")

    importance_matrix = np.array(importance_matrix)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Heatmapa
    im = ax.imshow(importance_matrix, cmap='RdBu_r',
                   aspect='auto', vmin=-0.2, vmax=0.5)

    # Ustaw etykiety
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Atomy (posortowane według ważności malejąco)', fontsize=12)
    ax.set_ylabel('Cząsteczka (CHEMBL ID)', fontsize=12)
    ax.set_title(
        'Heatmapa ważności atomów dla analizowanych cząsteczek', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ważność atomu (Integrated Gradients)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano heatmapę: {save_path}")
    plt.show()


def plot_atom_type_importance(results, save_path=None, top_k=5):
    """
    Analizuje, które typy atomów najczęściej pojawiają się wśród najważniejszych atomów.

    Args:
        results: lista wyników z analyze_dataset
        save_path: ścieżka do zapisu
        top_k: liczba najważniejszych atomów do analizy dla każdej cząsteczki
    """
    from collections import Counter

    # Podziel cząsteczki na dwie grupy
    # niskie pKpuu = dobre
    good_penetration = [r for r in results if r['true'] < 0]
    # wysokie pKpuu = słabe
    bad_penetration = [r for r in results if r['true'] >= 0]

    # Zliczaj typy atomów wśród top_k najważniejszych atomów
    good_counter = Counter()
    bad_counter = Counter()

    for res in good_penetration:
        mol = Chem.MolFromSmiles(res['smiles'])
        if mol is None:
            continue
        importance = res['importance']
        top_indices = np.argsort(importance)[-top_k:][::-1]
        for idx in top_indices:
            atom = mol.GetAtomWithIdx(int(idx))
            atom_type = atom.GetSymbol()
            good_counter[atom_type] += 1

    for res in bad_penetration:
        mol = Chem.MolFromSmiles(res['smiles'])
        if mol is None:
            continue
        importance = res['importance']
        top_indices = np.argsort(importance)[-top_k:][::-1]
        for idx in top_indices:
            atom = mol.GetAtomWithIdx(int(idx))
            atom_type = atom.GetSymbol()
            bad_counter[atom_type] += 1

    # Przygotuj dane do wykresu
    all_atom_types = set(good_counter.keys()) | set(bad_counter.keys())
    # Posortuj atomy według sumy wystąpień
    atom_types = sorted(all_atom_types,
                        key=lambda x: good_counter.get(
                            x, 0) + bad_counter.get(x, 0),
                        reverse=True)

    good_counts = [good_counter.get(atom, 0) for atom in atom_types]
    bad_counts = [bad_counter.get(atom, 0) for atom in atom_types]

    # Wykres
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(atom_types))
    width = 0.35

    bars1 = ax.bar(x - width/2, good_counts, width, label='Dobre przenikanie (niskie pKpuu)',
                   color='#2E86AB', alpha=0.7)
    bars2 = ax.bar(x + width/2, bad_counts, width, label='Słabe przenikanie (wysokie pKpuu)',
                   color='#F18F01', alpha=0.7)

    ax.set_xlabel('Typ atomu', fontsize=12)
    ax.set_ylabel(
        f'Liczba wystąpień wśród {top_k} najważniejszych atomów', fontsize=12)
    ax.set_title(f'Jakie atomy są najważniejsze dla przenikania przez BBB?\n'
                 f'(Analiza {top_k} najważniejszych atomów w każdej cząsteczce)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(atom_types, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Dodaj wartości na słupkach
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano wykres typów atomów: {save_path}")
    plt.show()

    # Dodatkowo: wypisz proporcje
    print("\n" + "="*50)
    print("ANALIZA TYPÓW ATOMÓW WŚRÓD NAJWAŻNIEJSZYCH")
    print("="*50)
    print(
        f"\nDobre przenikanie (niskie pKpuu) - łącznie {sum(good_counter.values())} wystąpień:")
    for atom in atom_types:
        count = good_counter.get(atom, 0)
        if count > 0:
            pct = count / sum(good_counter.values()) * 100
            print(f"  {atom}: {count} ({pct:.1f}%)")

    print(
        f"\nSłabe przenikanie (wysokie pKpuu) - łącznie {sum(bad_counter.values())} wystąpień:")
    for atom in atom_types:
        count = bad_counter.get(atom, 0)
        if count > 0:
            pct = count / sum(bad_counter.values()) * 100
            print(f"  {atom}: {count} ({pct:.1f}%)")


def plot_individual_atom_importance(results, save_path=None):
    """
    Dla KAŻDEJ cząsteczki pokazuje, które TYPY atomów są ważne.
    To jest alternatywa dla heatmapy - każda cząsteczka ma swój własny wykres słupkowy.
    """
    # Podziel na dobre i słabe przenikanie
    sorted_results = sorted(results, key=lambda x: x['true'])
    good = sorted_results[:5]   # 5 najlepiej przenikających
    bad = sorted_results[-5:]   # 5 najgorzej przenikających

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for i, res in enumerate(good):
        mol = Chem.MolFromSmiles(res['smiles'])
        if mol is None:
            continue

        importance = res['importance']

        # Zbierz ważność dla każdego typu atomu (uśredniona)
        atom_importance = {}
        atom_counts = {}

        for atom_idx, imp in enumerate(importance):
            atom = mol.GetAtomWithIdx(int(atom_idx))
            atom_type = atom.GetSymbol()
            if atom_type not in atom_importance:
                atom_importance[atom_type] = 0
                atom_counts[atom_type] = 0
            atom_importance[atom_type] += imp
            atom_counts[atom_type] += 1

        # Uśrednij
        for atom_type in atom_importance:
            atom_importance[atom_type] /= atom_counts[atom_type]

        # Posortuj według ważności
        sorted_atoms = sorted(atom_importance.items(),
                              key=lambda x: x[1], reverse=True)
        atom_types = [a[0] for a in sorted_atoms]
        avg_importance = [a[1] for a in sorted_atoms]

        colors = ['green' if val > 0 else 'red' for val in avg_importance]

        axes[0, i].bar(atom_types, avg_importance, color=colors, alpha=0.7)
        axes[0, i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, i].set_title(
            f"{res['chembl_id']}\npKpuu={res['true']:.2f}", fontsize=9)
        axes[0, i].tick_params(axis='x', rotation=45)
        axes[0, i].set_ylim(-0.3, 0.6)

        if i == 0:
            axes[0, i].set_ylabel('Średnia ważność atomu', fontsize=10)

    for i, res in enumerate(bad):
        mol = Chem.MolFromSmiles(res['smiles'])
        if mol is None:
            continue

        importance = res['importance']

        atom_importance = {}
        atom_counts = {}

        for atom_idx, imp in enumerate(importance):
            atom = mol.GetAtomWithIdx(int(atom_idx))
            atom_type = atom.GetSymbol()
            if atom_type not in atom_importance:
                atom_importance[atom_type] = 0
                atom_counts[atom_type] = 0
            atom_importance[atom_type] += imp
            atom_counts[atom_type] += 1

        for atom_type in atom_importance:
            atom_importance[atom_type] /= atom_counts[atom_type]

        sorted_atoms = sorted(atom_importance.items(),
                              key=lambda x: x[1], reverse=True)
        atom_types = [a[0] for a in sorted_atoms]
        avg_importance = [a[1] for a in sorted_atoms]

        colors = ['green' if val > 0 else 'red' for val in avg_importance]

        axes[1, i].bar(atom_types, avg_importance, color=colors, alpha=0.7)
        axes[1, i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, i].set_title(
            f"{res['chembl_id']}\npKpuu={res['true']:.2f}", fontsize=9)
        axes[1, i].tick_params(axis='x', rotation=45)
        axes[1, i].set_ylim(-0.3, 0.6)

        if i == 0:
            axes[1, i].set_ylabel('Średnia ważność atomu', fontsize=10)

    plt.suptitle('Średnia ważność dla poszczególnych typów atomów',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(
            f"Zapisano wykres typów atomów dla poszczególnych cząsteczek: {save_path}")
    plt.show()


if __name__ == "__main__":
    from model import MoleculePropertyPredictor
    from graph_utils import smiles_to_graph
    import os

    # Utwórz katalog na wizualizacje
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)

    # Ustaw urządzenie
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używam urządzenia: {device}")

    # Załaduj wytrenowany model GraphDenseNet
    model = MoleculePropertyPredictor().to(device)

    model_path = 'models_comparison/GraphDenseNet_best.pt'
    print(f"Ładowanie modelu z: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Analiza przykładowych cząsteczek
    results = analyze_dataset(
        model, device, 'data/raw/train_kpuu_log.csv', n_samples=10
    )

    # Zapisz wyniki do CSV
    if results:
        df_results = pd.DataFrame([{
            'chembl_id': r['chembl_id'],
            'smiles': r['smiles'],
            'true_pkpuu': r['true'],
            'pred_pkpuu': r['pred'],
            'error': abs(r['true'] - r['pred']),
            'num_atoms': r['num_atoms'],
            'mean_importance': r['importance'].mean(),
            'max_importance': r['importance'].max()
        } for r in results])
        df_results.to_csv('results/interpretability_results.csv', index=False)
        print("\nWyniki zapisane do interpretability_results.csv")

        # 1. Wykres słupkowy – najważniejsze atomy dla wybranych cząsteczek
        plot_atom_importance_for_molecules(
            results, save_path=os.path.join(viz_dir, 'atom_importance_bars.png'), top_k=6
        )

        # 2. Wykres porównawczy – wszystkie atomy, dobre vs słabe przenikanie
        plot_all_atoms_importance(
            results, save_path=os.path.join(
                viz_dir, 'atom_importance_comparison.png')
        )

        # 3. Heatmapa – macierz ważności
        plot_atom_importance_heatmap(
            results, save_path=os.path.join(
                viz_dir, 'atom_importance_heatmap.png')
        )

        # 4. Jakie typy atomów są najważniejsze?
        plot_atom_type_importance(
            results,
            save_path=os.path.join(viz_dir, 'atom_type_importance.png'),
            top_k=5
        )

        # 5. Dla każdej cząsteczki osobno – które typy atomów są ważne
        plot_individual_atom_importance(
            results,
            save_path=os.path.join(viz_dir, 'individual_atom_importance.png')
        )
