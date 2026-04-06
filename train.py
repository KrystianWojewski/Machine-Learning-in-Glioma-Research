import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from model import MoleculePropertyPredictor
from data_loader import create_dataloaders
from config import *

# Ustaw styl wykresów dla lepszego wyglądu
plt.style.use('seaborn-v0_8-darkgrid')
# Lub jeśli powyższy nie działa, użyj:
# plt.style.use('ggplot')

"""
Zatrzymuje trenowanie, gdy model przestaje się poprawiać.
"""

MODEL_DIR = ''


class EarlyStopping:

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


"""Trenowanie przez jedną epokę."""


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    for batch_graphs, labels in train_loader:
        batch_graphs = batch_graphs.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_graphs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    r2 = r2_score(targets, predictions)

    return avg_loss, r2


"""Ocena modelu na zbiorze danych."""


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_graphs, labels in loader:
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


"""
Rysuje krzywe uczenia się: loss i R² dla treningu i walidacji.

Args:
    history: słownik z historią trenowania
    model_name: nazwa modelu (dla tytułu wykresu i nazwy pliku)
    save_dir: katalog do zapisu wykresów
"""


def plot_learning_curves(history, target_col, save_dir=MODEL_DIR):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Wykres 1: Loss (MSE)
    axes[0].plot(epochs, history['train_loss'],
                 'b-', label='Trening', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-',
                 label='Walidacja', linewidth=2)
    axes[0].set_xlabel('Epoka', fontsize=12)
    axes[0].set_ylabel('Strata (MSE)', fontsize=12)
    axes[0].set_title(
        f'Krzywa uczenia się - {target_col}\n(Strata)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Znajdź najlepszą epokę (najniższa strata walidacyjna)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'Najlepszy model (epoka {best_epoch})')
    axes[0].text(best_epoch + 1, best_val_loss, f'  {best_val_loss:.4f}',
                 fontsize=9, color='green')

    # Wykres 2: R²
    axes[1].plot(epochs, history['train_r2'], 'b-',
                 label='Trening', linewidth=2)
    axes[1].plot(epochs, history['val_r2'], 'r-',
                 label='Walidacja', linewidth=2)
    axes[1].set_xlabel('Epoka', fontsize=12)
    axes[1].set_ylabel('Współczynnik determinacji (R²)', fontsize=12)
    axes[1].set_title(
        f'Krzywa uczenia się - {target_col}\n(Współczynnik R²)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Dodaj linię dla najlepszego R² walidacyjnego
    best_val_r2 = max(history['val_r2'])
    axes[1].axhline(y=best_val_r2, color='g', linestyle='--', alpha=0.5,
                    label=f'Najlepsze R² = {best_val_r2:.4f}')
    axes[1].legend(fontsize=10)

    plt.tight_layout()

    # Zapisz wykres
    save_path = os.path.join(save_dir, f'{target_col}_learning_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano krzywe uczenia się: {save_path}")
    plt.show()


"""
    Rysuje wykres przewidywanych vs rzeczywistych wartości.
    
    Args:
        test_true: rzeczywiste wartości
        test_pred: przewidywane wartości
        model_name: nazwa modelu
        save_dir: katalog do zapisu
    """


def plot_predictions_vs_true(test_true, test_pred, target_col, save_dir=MODEL_DIR):
    plt.figure(figsize=(8, 8))

    # Punkty
    plt.scatter(test_true, test_pred, alpha=0.5, c='steelblue', s=50)

    # Linia idealnej predykcji (y = x)
    min_val = min(min(test_true), min(test_pred))
    max_val = max(max(test_true), max(test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Idealna predykcja (y = x)')

    # Linia regresji
    z = np.polyfit(test_true, test_pred, 1)
    p = np.poly1d(z)
    plt.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2,
             label=f'Linia regresji (y = {z[0]:.2f}x + {z[1]:.2f})')

    plt.xlabel(f'Rzeczywiste {target_col}', fontsize=12)
    plt.ylabel(f'Przewidywane {target_col}', fontsize=12)
    plt.title(f'{target_col}\nPrzewidywane vs rzeczywiste wartości', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Dodaj metryki na wykresie
    r2 = r2_score(test_true, test_pred)
    rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    mae = mean_absolute_error(test_true, test_pred)

    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{target_col}_predictions_vs_true.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zapisano wykres przewidywań: {save_path}")
    plt.show()


def train(csv_file, target_col='Kpuu', model_name='model'):
    print(f"\n{'='*60}")
    print(f"Trenowanie modelu dla {target_col}")
    print(f"{'='*60}\n")

    # 1. Przygotowanie danych
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_file, target_col=target_col, batch_size=BATCH_SIZE
    )

    # 2. Inicjalizacja modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUżywam urządzenia: {device}")

    model = MoleculePropertyPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=EARLY_STOP)

    # 3. Pętla treningowa
    print("\nRozpoczynam trening...")
    print(f"Epoki: {EPOCHS}, Early stopping: {EARLY_STOP}\n")

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_r2': [],
        'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []
    }

    for epoch in range(EPOCHS):
        train_loss, train_r2 = train_epoch(
            model, train_loader, optimizer, criterion, device)

        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        # Zmniejsz learning rate
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train R²: {train_r2:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | "
                  f"Val R²: {val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                MODEL_DIR, f'{model_name}_best.pt'))

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping w epoce {epoch+1}")
            break

    # 4. Ewaluacja na zbiorze testowym
    print("\n" + "="*60)
    print("Ewaluacja na zbiorze testowym")
    print("="*60)

    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'{model_name}_best.pt')))
    test_loss, test_rmse, test_mae, test_r2, test_pred, test_true = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nWyniki testowe:")
    print(f"  MSE:  {test_loss:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # 5. Zapisz wyniki
    results_df = pd.DataFrame({
        'true': test_true,
        'predicted': test_pred
    })
    results_df.to_csv(os.path.join(
        MODEL_DIR, f'{target_col}_results.csv'), index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(
        MODEL_DIR, f'{target_col}_history.csv'), index=False)

    # 6. Narysuj krzywe uczenia się
    plot_learning_curves(history, target_col, save_dir=MODEL_DIR)

    # 7. Narysuj wykres przewidywane vs rzeczywiste
    plot_predictions_vs_true(test_true, test_pred,
                             target_col, save_dir=MODEL_DIR)

    print(f"\nWyniki zapisane w {MODEL_DIR}")

    return model, history, (test_rmse, test_mae, test_r2)


"""
Args:
    csv_file: ścieżka do pliku CSV z danymi
    target_col: kolumna do przewidzenia ('Kpuu' lub 'Kp')
    model_name: nazwa do zapisu modelu
"""

if __name__ == "__main__":
    csv_file = KPUU_FILE
    target_col = 'Kpuu'
    model_name = 'Kpuu_model'

    # Ustaw folder zapisu
    MODEL_DIR = os.path.join(
        RESULTS_DIR, f'{target_col}, Epochs = {EPOCHS}, EarlyStop = {EARLY_STOP}, LR = {LEARNING_RATE}')
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_kpuu, history_kpuu, metrics_kpuu = train(
        csv_file, target_col, model_name
    )

    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)
    print(
        f"{target_col} - RMSE: {metrics_kpuu[0]:.4f}, MAE: {metrics_kpuu[1]:.4f}, R²: {metrics_kpuu[2]:.4f}")
