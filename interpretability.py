import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import argparse
import os

from models import get_model
from graph_utils import smiles_to_graph

# ============================================================================
# FUNKCJE (integrated_gradients, plot_*, itd.)
# ============================================================================


def integrated_gradients(model, data, target_idx=0, steps=50):
    """
    Oblicza zintegrowane gradienty dla każdego atomu.
    """
    model.eval()

    device = next(model.parameters()).device
    data = data.to(device)

    baseline = torch.zeros_like(data.x)

    alphas = torch.linspace(0, 1, steps, device=device)
    interpolated = baseline + alphas.view(-1, 1, 1) * (data.x - baseline)

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

    grads = torch.stack(grads)
    avg_grads = torch.trapz(grads, alphas, dim=0)

    attributions = (data.x - baseline) * avg_grads
    node_importance = -attributions.sum(dim=1).detach().cpu().numpy()

    return node_importance


def get_atom_contributions(smiles, model, device):
    graph_dict = smiles_to_graph(smiles)
    if graph_dict is None:
        return None, None

    data = Data(
        x=graph_dict['x'],
        edge_index=graph_dict['edge_index'],
        edge_attr=graph_dict['edge_attr'],
        batch=torch.zeros(graph_dict['x'].shape[0], dtype=torch.long)
    ).to(device)

    importance = integrated_gradients(model, data)
    mol = Chem.MolFromSmiles(smiles)

    return mol, importance


def visualize_important_atoms(smiles, model, device, save_path=None):
    mol, importance = get_atom_contributions(smiles, model, device)

    if mol is None:
        print(f"Nie można przetworzyć SMILES: {smiles}")
        return

    print(f"\nCząsteczka: {smiles[:80]}...")
    print("Najważniejsze atomy (indeks, ważność):")
    top_indices = np.argsort(importance)[-5:][::-1]
    for idx in top_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        print(f"  Atom {int(idx)} ({atom.GetSymbol()}): {importance[idx]:.4f}")

    if save_path:
        highlight_atoms = [int(i) for i in top_indices[:3]]
        img = Draw.MolToImage(mol, size=(400, 400),
                              highlightAtoms=highlight_atoms,
                              highlightColor=(0.8, 0.2, 0.2))
        img.save(save_path)
        print(f"Zapisano wizualizację: {save_path}")

    return mol, importance


def analyze_dataset(model, device, csv_file, target_col, n_samples=10):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=[target_col])

    df_sorted = df.sort_values(target_col)
    low_samples = df_sorted.head(n_samples // 2)
    high_samples = df_sorted.tail(n_samples // 2)
    samples = pd.concat([low_samples, high_samples])

    print("\n" + "="*60)
    print(f"ANALIZA PRZYKŁADÓW - IDENTYFIKACJA WAŻNYCH ATOMÓW ({target_col})")
    print("="*60)

    results = []
    for idx, row in samples.iterrows():
        smiles = row['SMILES']
        true_value = row[target_col]
        chembl_id = row.get('CompoundID', f'CHEMBL_{idx}')

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

        model.eval()
        with torch.no_grad():
            pred = model(data).cpu().numpy()[0]

        importance = integrated_gradients(model, data)

        results.append({
            'chembl_id': chembl_id,
            'smiles': smiles,
            'true': true_value,
            'pred': pred,
            'importance': importance,
            'num_atoms': len(importance),
            'target_col': target_col
        })

        print(
            f"\n--- {chembl_id} ({target_col}: true={true_value:.3f}, pred={pred:.3f}) ---")
        print(f"SMILES: {smiles[:80]}...")
        print(f"Liczba atomów: {len(importance)}")
        print(f"Średnia ważność: {importance.mean():.4f}")
        print(f"Max ważność: {importance.max():.4f}")

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
    sorted_results = sorted(results, key=lambda x: x['true'])
    low_pkpuu = sorted_results[:2]
    high_pkpuu = sorted_results[-2:]
    selected = low_pkpuu + high_pkpuu

    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for i, res in enumerate(selected):
        importance = res['importance']
        mol = Chem.MolFromSmiles(res['smiles'])

        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_atoms = []
        top_importance = []

        for idx in top_indices:
            atom = mol.GetAtomWithIdx(int(idx))
            top_atoms.append(f"{atom.GetSymbol()}{int(idx)}")
            top_importance.append(importance[idx])

        colors = ['green' if val > 0 else 'red' for val in top_importance]

        axes[i].barh(top_atoms, top_importance, color=colors, alpha=0.7)
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].set_xlabel('Ważność atomu (Integrated Gradients)', fontsize=10)

        if res['true'] < 0:
            penetration = "DOBRE przenikanie"
        else:
            penetration = "SŁABE przenikanie"

        axes[i].set_title(f"{res['chembl_id']} | {penetration}\n"
                          f"{res.get('target_col', 'value')}: true={res['true']:.2f}, pred={res['pred']:.2f} | "
                          f"błąd={abs(res['true']-res['pred']):.2f}", fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano wykres ważności atomów: {save_path}")
    plt.show()


def plot_all_atoms_importance(results, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    all_importances = []
    all_types = []

    for res in results:
        importance = res['importance']
        is_good_penetration = res['true'] < 0

        for imp in importance:
            all_importances.append(imp)
            all_types.append(
                'Dobre przenikanie' if is_good_penetration else 'Słabe przenikanie')

    df_plot = pd.DataFrame({'importance': all_importances, 'type': all_types})

    positions = [0, 1]
    colors = ['#2E86AB', '#F18F01']

    parts = ax.violinplot([df_plot[df_plot['type'] == 'Dobre przenikanie']['importance'],
                           df_plot[df_plot['type'] == 'Słabe przenikanie']['importance']],
                          positions=positions, showmeans=False, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.5)

    for i, (type_name, color) in enumerate(zip(['Dobre przenikanie', 'Słabe przenikanie'], colors)):
        data = df_plot[df_plot['type'] == type_name]['importance']
        x_jitter = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x_jitter, data, alpha=0.3, s=10, color=color)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Próg (0)')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Dobre przenikanie\n(niskie {res.get('target_col', 'value')})',
                       f'Słabe przenikanie\n(wysokie {res.get('target_col', 'value')})'])
    ax.set_ylabel('Ważność atomu (Integrated Gradients)', fontsize=12)
    ax.set_title(
        'Porównanie ważności atomów\nw cząsteczkach dobrze i słabo przenikających przez BBB', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

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


def plot_individual_atom_importance(results, save_path=None):
    sorted_results = sorted(results, key=lambda x: x['true'])
    good = sorted_results[:5]
    bad = sorted_results[-5:]

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for i, res in enumerate(good):
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

        axes[0, i].bar(atom_types, avg_importance, color=colors, alpha=0.7)
        axes[0, i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, i].set_title(
            f"{res['chembl_id']}\n{res.get('target_col', 'value')}={res['true']:.2f}", fontsize=9)
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
            f"{res['chembl_id']}\n{res.get('target_col', 'value')}={res['true']:.2f}", fontsize=9)
        axes[1, i].tick_params(axis='x', rotation=45)
        axes[1, i].set_ylim(-0.3, 0.6)

        if i == 0:
            axes[1, i].set_ylabel('Średnia ważność atomu', fontsize=10)

    plt.suptitle('Średnia ważność dla poszczególnych typów atomów',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano wykres typów atomów: {save_path}")
    plt.show()


def plot_best_vs_worst_penetrator(results, save_path=None):
    if len(results) < 2:
        print("Za mało wyników do porównania")
        return

    sorted_results = sorted(results, key=lambda x: x['true'])
    best = sorted_results[0]
    worst = sorted_results[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Najlepszy przenikacz - wykres słupkowy
    importance_best = best['importance']
    mol_best = Chem.MolFromSmiles(best['smiles'])

    top_k = 8
    top_indices_best = np.argsort(importance_best)[-top_k:][::-1]
    top_atoms_best = []
    top_importance_best = []

    for idx in top_indices_best:
        atom = mol_best.GetAtomWithIdx(int(idx))
        top_atoms_best.append(f"{atom.GetSymbol()}{int(idx)}")
        top_importance_best.append(importance_best[idx])

    colors_best = ['green' if val >
                   0 else 'red' for val in top_importance_best]

    axes[0, 0].barh(top_atoms_best, top_importance_best,
                    color=colors_best, alpha=0.7)
    axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 0].set_xlabel('Ważność atomu', fontsize=10)
    axes[0, 0].set_title(
        f"NAJLEPSZY PRZENIKACZ\n{best['chembl_id']}\n{best.get('target_col', 'value')} = {best['true']:.3f}", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Najlepszy przenikacz - wzór chemiczny
    highlight_atoms_best = [int(i) for i in top_indices_best[:5]]
    img_best = Draw.MolToImage(mol_best, size=(300, 300),
                               highlightAtoms=highlight_atoms_best,
                               highlightColor=(0.8, 0.2, 0.2))
    axes[0, 1].imshow(img_best)
    axes[0, 1].axis('off')
    axes[0, 1].set_title(
        "Struktura cząsteczki\n(czerwone = najważniejsze atomy)", fontsize=10)

    # Najgorszy przenikacz - wykres słupkowy
    importance_worst = worst['importance']
    mol_worst = Chem.MolFromSmiles(worst['smiles'])

    top_indices_worst = np.argsort(importance_worst)[-top_k:][::-1]
    top_atoms_worst = []
    top_importance_worst = []

    for idx in top_indices_worst:
        atom = mol_worst.GetAtomWithIdx(int(idx))
        top_atoms_worst.append(f"{atom.GetSymbol()}{int(idx)}")
        top_importance_worst.append(importance_worst[idx])

    colors_worst = ['green' if val >
                    0 else 'red' for val in top_importance_worst]

    axes[1, 0].barh(top_atoms_worst, top_importance_worst,
                    color=colors_worst, alpha=0.7)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Ważność atomu', fontsize=10)
    axes[1, 0].set_title(
        f"NAJGORSZY PRZENIKACZ\n{worst['chembl_id']}\n{worst.get('target_col', 'value')} = {worst['true']:.3f}", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Najgorszy przenikacz - wzór chemiczny
    highlight_atoms_worst = [int(i) for i in top_indices_worst[:5]]
    img_worst = Draw.MolToImage(mol_worst, size=(300, 300),
                                highlightAtoms=highlight_atoms_worst,
                                highlightColor=(0.8, 0.2, 0.2))
    axes[1, 1].imshow(img_worst)
    axes[1, 1].axis('off')
    axes[1, 1].set_title(
        "Struktura cząsteczki\n(czerwone = najważniejsze atomy)", fontsize=10)

    plt.suptitle('Porównanie najlepszego i najgorszego przenikacza przez BBB',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Zapisano porównanie: {save_path}")
    plt.show()

    print("\n" + "="*60)
    print("PORÓWNANIE SKRAJNYCH PRZYPADKÓW")
    print("="*60)
    print(f"\nNAJLEPSZY PRZENIKACZ:")
    print(f"  ID: {best['chembl_id']}")
    print(f"  {best.get('target_col', 'value')}: {best['true']:.3f}")
    print(f"  Najważniejsze atomy:")
    for idx, (atom_name, val) in enumerate(zip(top_atoms_best[:5], top_importance_best[:5])):
        print(
            f"    {atom_name}: {val:.4f} ({'zwiększa' if val > 0 else 'zmniejsza'} przenikanie)")

    print(f"\nNAJGORSZY PRZENIKACZ:")
    print(f"  ID: {worst['chembl_id']}")
    print(f"  {worst.get('target_col', 'value')}: {worst['true']:.3f}")
    print(f"  Najważniejsze atomy:")
    for idx, (atom_name, val) in enumerate(zip(top_atoms_worst[:5], top_importance_worst[:5])):
        print(
            f"    {atom_name}: {val:.4f} ({'zwiększa' if val > 0 else 'zmniejsza'} przenikanie)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Interpretowalność modeli do predykcji Kp/Kpuu')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Ścieżka do wytrenowanego modelu (best_model.pt)')
    parser.add_argument('--csv', type=str, required=True,
                        help='Plik CSV z danymi (np. data/raw/train_kpuu_log.csv)')
    parser.add_argument('--target_col', type=str, default='pKpuu',
                        help='Kolumna docelowa (pKpuu lub pKp)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Liczba próbek do analizy')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUżywam urządzenia: {device}")
    print(f"Plik danych: {args.csv}")
    print(f"Kolumna docelowa: {args.target_col}")
    print(f"Ścieżka modelu: {args.model_path}")

    if 'GCN' in args.model_path:
        model_type = 'GCN'
    elif 'GIN' in args.model_path:
        model_type = 'GIN'
    else:
        model_type = 'GraphDenseNet'

    print(f"Typ modelu: {model_type}")

    use_descriptors = False

    if use_descriptors:
        from descriptors import get_descriptor_names
        desc_dim = len(get_descriptor_names())
    else:
        desc_dim = None

    model = get_model(model_type, descriptor_dim=desc_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    save_dir = f'results/{args.model_path.split("\\")[1]}/interpretability'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Katalog zapisu: {save_dir}")

    results = analyze_dataset(model, device, args.csv,
                              args.target_col, args.n_samples)

    if results:
        df_results = pd.DataFrame([{
            'chembl_id': r['chembl_id'],
            'smiles': r['smiles'],
            'true_value': r['true'],
            'pred_value': r['pred'],
            'error': abs(r['true'] - r['pred']),
            'num_atoms': r['num_atoms'],
            'mean_importance': r['importance'].mean(),
            'max_importance': r['importance'].max()
        } for r in results])
        df_results.to_csv(os.path.join(
            save_dir, 'interpretability_results.csv'), index=False)
        print(f"\nWyniki zapisane w {save_dir}")

        plot_atom_importance_for_molecules(
            results, save_path=os.path.join(save_dir, 'atom_importance_bars.png'), top_k=6
        )
        plot_all_atoms_importance(
            results, save_path=os.path.join(
                save_dir, 'atom_importance_comparison.png')
        )
        plot_individual_atom_importance(
            results, save_path=os.path.join(
                save_dir, 'individual_atom_importance.png')
        )
        plot_best_vs_worst_penetrator(
            results, save_path=os.path.join(save_dir, 'best_vs_worst.png')
        )

"""
Args:
--model_path: Ścieżka do wytrenowanego modelu (np. results/TL_GIN_kpuu_to_pKpuu_100to150_1e-4to1e-5_32to16_0.5/best_model.pt)
--csv: Plik CSV z danymi (np. data/raw/train_kpuu_log.csv)
--target_col: Kolumna docelowa (pKpuu lub pKp)
--n_samples: Liczba próbek do analizy (domyślnie 10)

py interpretability.py --model_path results/GraphDenseNet_pKpuu_E400_LR0.001_BS32/best_model.pt --csv data/raw/train_kpuu_log.csv --target_col pKpuu
py interpretability.py --model_path results/GraphDenseNet_pKp_E200_LR0.0005_BS32/best_model.pt --csv data/raw/train_kp_log.csv --target_col pKp

"""
