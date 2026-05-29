"""
Trenowanie modeli do predykcji Kp/Kpuu.
Obsługa argumentów wiersza poleceń.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from models import (
    GCNModel, GINModel, GraphDenseNetModel,
    DescriptorOnlyModel, HybridModel
)
from data_loader import create_dataloaders

# Ustaw styl wykresów
plt.style.use('seaborn-v0_8-darkgrid')


class EarlyStopping:
    """Zatrzymuje trenowanie, gdy model przestaje się poprawiać."""

    def __init__(self, patience=50, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, optimizer, criterion, device, model_type):
    """Trenowanie przez jedną epokę."""
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    for batch in train_loader:
        # Hybrid: (graph, descriptors, labels)
        if model_type in ['Hybrid', 'hybrid']:
            batch_graphs, batch_descs, labels = batch
            batch_graphs = batch_graphs.to(device)
            batch_descs = batch_descs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_graphs, batch_descs)

        # Descriptor only: (graph, descriptors, labels) - pomijamy graf
        elif model_type in ['DescriptorOnly', 'Descriptor', 'descriptor']:
            _, batch_descs, labels = batch
            batch_descs = batch_descs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_descs)

        # Graph only (GCN, GIN, GraphDenseNet)
        else:
            batch_graphs, labels = batch
            batch_graphs = batch_graphs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_graphs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    r2 = r2_score(targets, predictions)

    return avg_loss, r2


def evaluate(model, loader, criterion, device, model_type):
    """Ocena modelu na zbiorze danych."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            if model_type in ['Hybrid', 'hybrid']:
                batch_graphs, batch_descs, labels = batch
                batch_graphs = batch_graphs.to(device)
                batch_descs = batch_descs.to(device)
                labels = labels.to(device)
                outputs = model(batch_graphs, batch_descs)

            elif model_type in ['DescriptorOnly', 'Descriptor', 'descriptor']:
                _, batch_descs, labels = batch
                batch_descs = batch_descs.to(device)
                labels = labels.to(device)
                outputs = model(batch_descs)

            else:  # Graph only
                batch_graphs, labels = batch
                batch_graphs = batch_graphs.to(device)
                labels = labels.to(device)
                outputs = model(batch_graphs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return avg_loss, rmse, mae, r2, predictions, targets


def plot_learning_curves(history, model_type, target_col, save_dir):
    """Rysuje krzywe uczenia się."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'],
                 'b-', label='Trening', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-',
                 label='Walidacja', linewidth=2)
    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Strata (MSE)', fontsize=12)
    axes[0].set_title(
        f'Krzywa uczenia się - {model_type}\n(Strata)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'Najlepszy model (epoka {best_epoch})')
    axes[0].text(best_epoch + 1, best_val_loss, f'  {best_val_loss:.4f}',
                 fontsize=9, color='green')

    axes[1].plot(epochs, history['train_r2'], 'b-',
                 label='Trening', linewidth=2)
    axes[1].plot(epochs, history['val_r2'], 'r-',
                 label='Walidacja', linewidth=2)
    axes[1].set_xlabel('Epoka', fontsize=12)
    axes[1].set_ylabel('Współczynnik determinacji (R²)', fontsize=12)
    axes[1].set_title(
        f'Krzywa uczenia się - {model_type}\n(Współczynnik R²)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    best_val_r2 = max(history['val_r2'])
    axes[1].axhline(y=best_val_r2, color='g', linestyle='--', alpha=0.5,
                    label=f'Najlepsze R² = {best_val_r2:.4f}')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'learning_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano krzywe uczenia się: {save_path}")
    plt.close()


def plot_predictions_vs_true(test_true, test_pred, model_type, target_col, save_dir):
    """Rysuje wykres przewidywanych vs rzeczywistych wartości."""
    plt.figure(figsize=(8, 8))

    plt.scatter(test_true, test_pred, alpha=0.5, c='steelblue', s=50)

    min_val = min(min(test_true), min(test_pred))
    max_val = max(max(test_true), max(test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Idealna predykcja (y = x)')

    z = np.polyfit(test_true, test_pred, 1)
    p = np.poly1d(z)
    plt.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2,
             label=f'Linia regresji (y = {z[0]:.2f}x + {z[1]:.2f})')

    plt.xlabel(f'Rzeczywiste {target_col}', fontsize=12)
    plt.ylabel(f'Przewidywane {target_col}', fontsize=12)
    plt.title(f'{model_type}\nPrzewidywane vs rzeczywiste wartości', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    r2 = r2_score(test_true, test_pred)
    rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    mae = mean_absolute_error(test_true, test_pred)

    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'predictions_vs_true.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano wykres przewidywań: {save_path}")
    plt.close()


def get_model(model_type, descriptor_dim=None):
    """Zwraca odpowiedni model na podstawie typu."""
    if model_type in ['GCN', 'gcn']:
        return GCNModel()
    elif model_type in ['GIN', 'gin']:
        return GINModel()
    elif model_type in ['GraphDenseNet', 'Graph', 'graph']:
        return GraphDenseNetModel()
    elif model_type in ['DescriptorOnly', 'Descriptor', 'descriptor']:
        return DescriptorOnlyModel(descriptor_dim=descriptor_dim)
    elif model_type in ['Hybrid', 'hybrid']:
        return HybridModel(descriptor_dim=descriptor_dim)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}. "
                         f"Dostępne: GCN, GIN, GraphDenseNet, DescriptorOnly, Hybrid")


def train(args, data_type, transform):
    """
    Główna funkcja trenowania.
    """
    # Złóż nazwę folderu
    folder_name = f"{args.model}_{data_type}_{transform}_E{args.epochs}_LR{args.lr}_BS{args.batch_size}"

    # Dodaj early stopping do nazwy jeśli nie jest domyślne
    if args.early_stop != 50:
        folder_name += f"_ES{args.early_stop}"

    save_dir = os.path.join('results', folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Trenowanie modelu {args.model} dla {args.target_col}")
    print(f"{'='*60}")
    print(f"\nParametry trenowania:")
    print(f"  Plik danych: {args.csv_file}")
    print(f"  Typ danych: {data_type}")
    print(f"  Transformacja: {transform}")
    print(f"  Cel predykcji: {args.target_col}")
    print(f"  Typ modelu: {args.model}")
    print(f"  Liczba epok: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Early stopping: {args.early_stop}")
    print(f"  Folder wyników: {save_dir}")

    # Przygotowanie danych - deskryptory tylko dla modeli, które ich potrzebują
    use_descriptors = args.model in ['DescriptorOnly', 'Descriptor', 'descriptor',
                                     'Hybrid', 'hybrid']

    train_loader, val_loader, test_loader = create_dataloaders(
        args.csv_file, target_col=args.target_col, batch_size=args.batch_size,
        use_descriptors=use_descriptors
    )

    # Inicjalizacja modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUżywam urządzenia: {device}")

    if use_descriptors:
        from descriptors import get_descriptor_names
        desc_dim = len(get_descriptor_names())
        model = get_model(args.model, descriptor_dim=desc_dim).to(device)
    else:
        model = get_model(args.model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.early_stop)

    print("\nRozpoczynam trening...")

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_r2': [],
        'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []
    }

    for epoch in range(args.epochs):
        # Trening
        train_loss, train_r2 = train_epoch(
            model, train_loader, optimizer, criterion, device, args.model
        )

        # Walidacja
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device, args.model
        )

        history['train_loss'].append(train_loss)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train R²: {train_r2:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val R²: {val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best_model.pt'))

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping w epoce {epoch+1}")
            break

    # Ewaluacja na zbiorze testowym
    print("\n" + "="*60)
    print("Ewaluacja na zbiorze testowym")
    print("="*60)

    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_rmse, test_mae, test_r2, test_pred, test_true = evaluate(
        model, test_loader, criterion, device, args.model
    )

    print(f"\nWyniki testowe:")
    print(f"  MSE:  {test_loss:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # Zapisz wyniki
    results_df = pd.DataFrame({
        'true': test_true,
        'predicted': test_pred
    })
    results_df.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(
        save_dir, 'training_history.csv'), index=False)

    # Zapisz metryki
    metrics_df = pd.DataFrame([{
        'model': args.model,
        'target_col': args.target_col,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'early_stop': args.early_stop,
        'test_loss': test_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'best_epoch': np.argmin(history['val_loss']) + 1
    }])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    # Narysuj wykresy
    plot_learning_curves(history, args.model, args.target_col, save_dir)
    plot_predictions_vs_true(test_true, test_pred,
                             args.model, args.target_col, save_dir)

    print(f"\nWyniki zapisane w {save_dir}")

    return model, history, (test_rmse, test_mae, test_r2)


def compare_models(args, data_type, transform):
    """
    Porównuje wszystkie modele z tymi samymi parametrami.
    Rysuje zaawansowane wykresy porównawcze.
    """
    models_to_test = ['GCN', 'GIN',
                      'GraphDenseNet', 'DescriptorOnly', 'Hybrid']
    results = {}
    histories = {}

    print("\n" + "="*70)
    print("PORÓWNANIE WSZYSTKICH MODELI")
    print("="*70)
    print(f"\nParametry:")
    print(f"  Plik danych: {args.csv_file}")
    print(f"  Cel predykcji: {args.target_col}")
    print(f"  Liczba epok: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Testowane modele: {', '.join(models_to_test)}")

    for model_type in models_to_test:
        print("\n" + "="*50)
        print(f"Trenowanie: {model_type}")
        print("="*50)

        # Stwórz nowy args dla każdego modelu
        model_args = argparse.Namespace(
            csv_file=args.csv_file,
            target_col=args.target_col,
            model=model_type,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            early_stop=args.early_stop
        )
        _, history, metrics = train(model_args, data_type, transform)
        results[model_type] = {
            'rmse': metrics[0],
            'mae': metrics[1],
            'r2': metrics[2]
        }
        histories[model_type] = history

    # ========================================================================
    # Zaawansowane wykresy porównawcze
    # ========================================================================

    # 1. Wykres słupkowy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#1B998B', '#E84855']

    # RMSE
    rmse_vals = [results[m]['rmse'] for m in model_names]
    bars = axes[0].bar(model_names, rmse_vals, color=colors[:len(
        model_names)], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title('Porównanie modeli - RMSE (niższy lepszy)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, rmse_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                     0.01, f'{val:.4f}', ha='center', fontsize=9)

    # MAE
    mae_vals = [results[m]['mae'] for m in model_names]
    bars = axes[1].bar(model_names, mae_vals, color=colors[:len(
        model_names)], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Porównanie modeli - MAE (niższy lepszy)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, mae_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                     0.01, f'{val:.4f}', ha='center', fontsize=9)

    # R²
    r2_vals = [results[m]['r2'] for m in model_names]
    bars = axes[2].bar(model_names, r2_vals, color=colors[:len(
        model_names)], alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].set_title('Porównanie modeli - R² (wyższy lepszy)', fontsize=12)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, r2_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                     0.01, f'{val:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join('results', f'all_models_comparison_{data_type}_{transform}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Krzywe uczenia się (wszystkie modele na jednym wykresie)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, history) in enumerate(histories.items()):
        epochs = range(1, len(history['val_loss']) + 1)
        axes[0].plot(epochs, history['val_loss'], label=name,
                     linewidth=2, color=colors[i % len(colors)])
        axes[1].plot(epochs, history['val_r2'], label=name,
                     linewidth=2, color=colors[i % len(colors)])

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
    plt.savefig(os.path.join(
        'results', f'all_models_learning_curves_{data_type}_{transform}.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Zapisz tabelę porównawczą
    comparison_df = pd.DataFrame([
        {'Model': name, 'RMSE': res['rmse'],
            'MAE': res['mae'], 'R²': res['r2']}
        for name, res in results.items()
    ])
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    comparison_df.to_csv(os.path.join(
        'results', f'all_models_comparison_{data_type}_{transform}.csv'), index=False)

    print("\n" + "="*70)
    print("PODSUMOWANIE PORÓWNANIA MODELI")
    print("="*70)
    print(comparison_df.to_string(index=False))

    return results, histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trenowanie modeli do predykcji Kp/Kpuu')

    # Argumenty danych
    parser.add_argument('--csv_file', type=str, default='data/raw/train_kpuu_log.csv',
                        help='Ścieżka do pliku CSV z danymi')
    parser.add_argument('--target_col', type=str, default='pKpuu',
                        choices=['Kpuu', 'Kp', 'pKpuu', 'pKp'],
                        help='Kolumna do przewidzenia')

    # Argumenty modelu
    parser.add_argument('--model', type=str, default='GraphDenseNet',
                        choices=['GCN', 'GIN', 'GraphDenseNet',
                                 'DescriptorOnly', 'Hybrid', 'Compare'],
                        help='Typ modelu (lub "Compare" do porównania wszystkich)')

    # Argumenty trenowania
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maksymalna liczba epok')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='Cierpliwość early stopping')

    args = parser.parse_args()

    # Wyciągnij nazwę pliku CSV (bez rozszerzenia i ścieżki)
    csv_filename = os.path.splitext(os.path.basename(args.csv_file))[0]

    # Przygotuj oznaczenie typu danych
    if 'kpuu' in csv_filename.lower():
        data_type = 'Kpuu'
    elif 'kp' in csv_filename.lower():
        data_type = 'KP'
    else:
        data_type = 'data'

    # Dodaj informację o transformacji logarytmicznej
    if args.target_col in ['pKpuu', 'pKp']:
        transform = 'log'
    else:
        transform = 'raw'

    # Uruchom odpowiednią funkcję
    if args.model == 'Compare':
        compare_models(args, data_type, transform)
    else:
        train(args, data_type, transform)

"""
Args:
--model GCN/GIN/GraphDenseNet/DescriptorOnly/Hybrid/Compare
--epochs
--lr (learning rate)
--batch_size
--early_stop (patience for early stopping)
--csv_file data/raw/train_kpuu_log.csv (ścieżka do pliku CSV)
--target_col (Kpuu/Kp/pKpuu/pKp)

py train.py --model GCN --epochs 200 --lr 0.0005 --batch_size 32 --early_stop 50 --csv_file data/raw/train_kpuu_log.csv --target_col pKpuu
"""
