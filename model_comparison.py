"""
Porównanie różnych architektur GNN do przewidywania Kpuu.
Testowane architektury:
1. GCN (Graph Convolutional Network) - baseline
2. GIN (Graph Isomorphism Network) - teoretycznie najsilniejsza
3. GraphDenseNet - nasza dotychczasowa architektura
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch.nn import BatchNorm1d

from config import MODELS_COMPARISON_DIR


# Ustaw styl wykresów
plt.style.use('seaborn-v0_8-darkgrid')
# Jeśli powyższy nie działa, użyj:
# plt.style.use('ggplot')


class GCNModel(nn.Module):
    """
    Graph Convolutional Network - prosta, klasyczna architektura.

    Zalety: szybka, mało parametrów
    Wady: mniej ekspresywna niż GIN
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

        # Głowa regresyjna
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Warstwy konwolucyjne
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

        # Pooling
        x = global_mean_pool(x, batch)

        # Regresja
        x = self.regressor(x)

        return x.view(-1)


class GINModel(nn.Module):
    """
    Graph Isomorphism Network - teoretycznie najsilniejsza architektura GNN.

    Dlaczego GIN jest teoretycznie silniejszy?
    - Może rozróżniać różne struktury grafów, które GCN myli
    - Ma teoretyczne gwarancje ekspresywności (tak silny jak 1-WL test)

    Zalety: bardzo ekspresywny, dobry dla małych grafów (jak cząsteczki)
    Wady: więcej parametrów, wolniejszy
    """

    def __init__(self, in_channels=74, hidden_dim=128, dropout=0.2):
        super().__init__()

        # GIN używa MLP zamiast prostej warstwy liniowej
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

        # Głowa regresyjna
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


def plot_comparison_results(results, save_dir=MODELS_COMPARISON_DIR):
    """
    Tworzy wizualizację porównania modeli.

    Args:
        results: słownik z wynikami dla każdego modelu
        save_dir: katalog do zapisu wykresów
    """

    # Przygotuj dane do wykresów
    model_names = list(results.keys())
    r2_scores = [results[name]['test_r2'] for name in model_names]
    rmse_scores = [results[name]['test_rmse'] for name in model_names]
    mae_scores = [results[name]['test_mae'] for name in model_names]
    best_epochs = [results[name]['best_epoch'] for name in model_names]

    # Kolory dla modeli
    # niebieski, fioletowy, pomarańczowy
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Wykres 1: R² (wyższy = lepszy)
    bars1 = axes[0, 0].bar(model_names, r2_scores,
                           color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Współczynnik determinacji (R²)', fontsize=12)
    axes[0, 0].set_title('Porównanie modeli - R²',
                         fontsize=12, fontweight='bold')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylim(-0.1, max(r2_scores) + 0.1)

    # Dodaj wartości na słupkach
    for bar, val in zip(bars1, r2_scores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Wykres 2: RMSE (niższy = lepszy)
    bars2 = axes[0, 1].bar(model_names, rmse_scores,
                           color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('RMSE (Root Mean Square Error)', fontsize=12)
    axes[0, 1].set_title('Porównanie modeli - RMSE',
                         fontsize=12, fontweight='bold')

    for bar, val in zip(bars2, rmse_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Wykres 3: MAE (niższy = lepszy)
    bars3 = axes[1, 0].bar(model_names, mae_scores,
                           color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
    axes[1, 0].set_title('Porównanie modeli - MAE',
                         fontsize=12, fontweight='bold')

    for bar, val in zip(bars3, mae_scores):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Wykres 4: Najlepsza epoka
    bars4 = axes[1, 1].bar(model_names, best_epochs,
                           color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Najlepsza epoka', fontsize=12)
    axes[1, 1].set_title('Porównanie modeli - Szybkość uczenia',
                         fontsize=12, fontweight='bold')

    for bar, val in zip(bars4, best_epochs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Zapisz wykres
    save_path = os.path.join(save_dir, 'models_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano porównanie modeli: {save_path}")
    plt.show()

    return fig


def plot_radar_comparison(results, save_dir=MODELS_COMPARISON_DIR):
    """
    Tworzy wykres radarowy do porównania modeli (znormalizowane metryki).

    Args:
        results: słownik z wynikami dla każdego modelu
        save_dir: katalog do zapisu wykresów
    """

    model_names = list(results.keys())

    # Metryki do porównania (normalizujemy, żeby były w skali 0-1)
    # Dla R²: im wyższy tym lepszy
    # Dla RMSE i MAE: im niższy tym lepszy
    r2_scores = [results[name]['test_r2'] for name in model_names]
    rmse_scores = [results[name]['test_rmse'] for name in model_names]
    mae_scores = [results[name]['test_mae'] for name in model_names]

    # Normalizacja (odwrócona dla RMSE i MAE)
    max_r2 = max(r2_scores)
    min_rmse = min(rmse_scores)
    min_mae = min(mae_scores)

    r2_norm = [r2 / max_r2 for r2 in r2_scores]
    rmse_norm = [min_rmse / rmse for rmse in rmse_scores]  # odwrotność
    mae_norm = [min_mae / mae for mae in mae_scores]       # odwrotność

    # Kąty dla osi radaru
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # zamknięcie wykresu

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Kolory dla modeli
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for i, name in enumerate(model_names):
        values = [r2_norm[i], rmse_norm[i], mae_norm[i]]
        values += values[:1]  # zamknięcie
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Ustaw etykiety osi
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        ['R² (wyższy lepszy)', 'RMSE (niższy lepszy)', 'MAE (niższy lepszy)'])
    ax.set_ylim(0, 1)
    ax.set_title('Porównanie modeli - Wykres radarowy\n(metryki znormalizowane)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'models_comparison_radar.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano wykres radarowy: {save_path}")
    plt.show()


def create_comparison_table(results, save_dir=MODELS_COMPARISON_DIR):
    """
    Tworzy tabelę porównawczą modeli i zapisuje jako CSV.

    Args:
        results: słownik z wynikami dla każdego modelu
        save_dir: katalog do zapisu
    """

    # Stwórz DataFrame
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'R²': [results[name]['test_r2'] for name in results.keys()],
        'RMSE': [results[name]['test_rmse'] for name in results.keys()],
        'MAE': [results[name]['test_mae'] for name in results.keys()],
        'Best Epoch': [results[name]['best_epoch'] for name in results.keys()]
    })

    # Sortuj według R² (malejąco)
    df = df.sort_values('R²', ascending=False)

    # Zapisz do CSV
    save_path = os.path.join(save_dir, 'models_comparison_table.csv')
    df.to_csv(save_path, index=False)
    print(f"Zapisano tabelę porównawczą: {save_path}")

    # Wyświetl tabelę w konsoli
    print("\n" + "="*60)
    print("TABELA PORÓWNAWCZA MODELI")
    print("="*60)
    print(df.to_string(index=False))

    return df


def train_all_models(csv_file, target_col='pKpuu', learning_rate=0.001):
    """
    Trenuje wszystkie trzy architektury i porównuje wyniki.
    """
    from data_loader import create_dataloaders
    from train import train_epoch, evaluate
    from model import MoleculePropertyPredictor
    import torch.optim as optim

    # Utwórz katalog dla wyników porównania
    os.makedirs(MODELS_COMPARISON_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("ANALIZA PORÓWNAWCZA ARCHITEKTUR GNN")
    print("="*70)

    # Przygotuj dane (te same dla wszystkich modeli)
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_file, target_col=target_col, batch_size=32
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUżywam urządzenia: {device}")

    # Definicje modeli do przetestowania
    models = {
        'GCN': GCNModel(),
        'GIN': GINModel(),
        'GraphDenseNet': MoleculePropertyPredictor()
    }

    results = {}

    # Słownik do przechowywania historii dla każdego modelu
    histories = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Trenowanie: {name}")
        print(f"{'='*50}")

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_epoch = 0

        # Historia dla tego modelu
        history = {
            'train_loss': [], 'train_r2': [],
            'val_loss': [], 'val_r2': []
        }

        # Trening (100 epok dla porównania)
        for epoch in range(100):
            # Trening
            train_loss, train_r2 = train_epoch(
                model, train_loader, optimizer, criterion, device)

            # Walidacja
            val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
                model, val_loader, criterion, device)

            # Zapisz historię
            history['train_loss'].append(train_loss)
            history['train_r2'].append(train_r2)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(
                    MODELS_COMPARISON_DIR, f'{name}_best.pt'))

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val R²={val_r2:.4f}")

        histories[name] = history

        # Ewaluacja na teście
        model.load_state_dict(torch.load(
            os.path.join(MODELS_COMPARISON_DIR, f'{name}_best.pt')))
        test_loss, test_rmse, test_mae, test_r2, test_pred, test_true = evaluate(
            model, test_loader, criterion, device)

        results[name] = {
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'best_epoch': best_epoch
        }

        print(f"\n{name} - Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        # Zapisz wyniki predykcji dla tego modelu
        pred_df = pd.DataFrame({
            'true': test_true,
            'predicted': test_pred
        })
        pred_df.to_csv(os.path.join(MODELS_COMPARISON_DIR,
                       f'{name}_predictions.csv'), index=False)

    # Wizualizacja wyników
    print("\n" + "="*70)
    print("TWORZENIE WIZUALIZACJI")
    print("="*70)

    # Tabela porównawcza
    comparison_table = create_comparison_table(
        results, save_dir=MODELS_COMPARISON_DIR)

    # Wykres słupkowy
    plot_comparison_results(results, save_dir=MODELS_COMPARISON_DIR)

    # Wykres radarowy
    plot_radar_comparison(results, save_dir=MODELS_COMPARISON_DIR)

    # Dodatkowo: wykres krzywych uczenia się dla każdego modelu
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'GCN': '#2E86AB', 'GIN': '#A23B72', 'GraphDenseNet': '#F18F01'}

    for name, history in histories.items():
        epochs = range(1, len(history['val_loss']) + 1)
        axes[0].plot(epochs, history['val_loss'], label=name,
                     color=colors[name], linewidth=2)
        axes[1].plot(epochs, history['val_r2'], label=name,
                     color=colors[name], linewidth=2)

    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Strata walidacyjna (MSE)', fontsize=12)
    axes[0].set_title('Porównanie krzywych uczenia się - Strata', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoka', fontsize=12)
    axes[1].set_ylabel('R² walidacyjny', fontsize=12)
    axes[1].set_title('Porównanie krzywych uczenia się - R²', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(MODELS_COMPARISON_DIR,
                             'models_learning_curves_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano porównanie krzywych uczenia się: {save_path}")
    plt.show()

    # Podsumowanie
    print("\n" + "="*70)
    print("PODSUMOWANIE PORÓWNANIA")
    print("="*70)
    print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Best Epoch'}")
    print("-"*50)
    for name, metrics in results.items():
        print(
            f"{name:<15} {metrics['test_r2']:<10.4f} {metrics['test_rmse']:<10.4f} {metrics['test_mae']:<10.4f} {metrics['best_epoch']}")

    return results, histories


if __name__ == "__main__":
    from model import MoleculePropertyPredictor

    # Upewnij się, że katalog istnieje
    os.makedirs(MODELS_COMPARISON_DIR, exist_ok=True)

    results, histories = train_all_models(
        'data/raw/train_kpuu_log.csv', target_col='pKpuu', learning_rate=0.001
    )
