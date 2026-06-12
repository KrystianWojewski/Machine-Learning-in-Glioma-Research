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
    DescriptorOnlyModel, HybridModel, ResidualGCNModel
)
from data_loader import create_dataloaders
from plot_utils import PlotGenerator

# Ustaw styl wykresów
plt.style.use('seaborn-v0_8-darkgrid')


def set_seed(seed):
    """Ustawia seed."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Zatrzymuje trenowanie, gdy model przestaje się poprawiać."""

    def __init__(self, patience=50, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


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


def get_model(model_type, descriptor_dim=None):
    """Zwraca odpowiedni model na podstawie typu."""
    if model_type in ['GCN', 'gcn']:
        return GCNModel()
    elif model_type in ['GIN', 'gin']:
        return GINModel()
    elif model_type in ['GraphDenseNet', 'Graph', 'graph']:
        return GraphDenseNetModel()
    elif model_type in ['ResidualGCN', 'residualgcn']:
        return ResidualGCNModel()
    elif model_type in ['DescriptorOnly', 'Descriptor', 'descriptor']:
        return DescriptorOnlyModel(descriptor_dim=descriptor_dim)
    elif model_type in ['Hybrid', 'hybrid']:
        return HybridModel(descriptor_dim=descriptor_dim)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}. "
                         f"Dostępne: GCN, GIN, GraphDenseNet, DescriptorOnly, Hybrid")


def train(args):
    """
    Główna funkcja trenowania.
    """
    # Ustaw seed
    set_seed(args.seed)
    # Złóż nazwę folderu
    folder_name = f"{args.model}_{args.target_col}_E{args.epochs}_LR{args.lr}_BS{args.bs}_SEED{args.seed}"

    save_dir = os.path.join('results', folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Trenowanie modelu {args.model} dla {args.target_col}")
    print(f"{'='*60}")
    print(f"\nParametry trenowania:")
    print(f"  Plik danych: {args.csv}")
    print(f"  Typ danych: {args.target_col}")
    print(f"  Typ modelu: {args.model}")
    print(f"  Liczba epok: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.bs}")
    print(f"  Early stopping: {args.es}")
    print(f"  Seed: {args.seed}")
    print(f"  Folder wyników: {save_dir}")

    # Przygotowanie danych - deskryptory tylko dla modeli, które ich potrzebują
    use_descriptors = args.model in ['DescriptorOnly', 'Descriptor', 'descriptor',
                                     'Hybrid', 'hybrid']

    train_loader, val_loader, test_loader = create_dataloaders(
        args.csv, target_col=args.target_col, batch_size=args.bs,
        use_descriptors=use_descriptors, random_state=args.seed, save_dir=save_dir
    )

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
    early_stopping = EarlyStopping(patience=args.es)

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

        early_stopping(val_loss, epoch+1)
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
        'seed': args.seed,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.bs,
        'early_stop': args.es,
        'test_loss': test_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'best_epoch': np.argmin(history['val_loss']) + 1
    }])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    # Narysuj wykresy
    plotter = PlotGenerator(save_dir=save_dir)
    plotter.plot_learning_curves(
        history, args.model, args.target_col, args.epochs, args.seed)
    plotter.plot_predictions_vs_true(
        test_true, test_pred, args.model, args.target_col)
    plotter.plot_residuals(test_true, test_pred, args.model, args.target_col)
    plotter.plot_errors_vs_target(
        test_true, test_pred, args.model, args.target_col)

    print(f"\nWyniki zapisane w {save_dir}")

    return model, history, (test_rmse, test_mae, test_r2)


def train_all_models(args):
    """
    Porównuje wszystkie modele z tymi samymi parametrami.
    Rysuje zaawansowane wykresy porównawcze.
    """
    models_to_test = ['GCN', 'GIN',
                      'GraphDenseNet', 'DescriptorOnly', 'Hybrid', 'ResidualGCN']
    results = {}
    histories = {}

    print("\n" + "="*60)
    print("PORÓWNANIE WSZYSTKICH MODELI")
    print("="*60)
    print(f"\nParametry:")
    print(f"  Plik danych: {args.csv}")
    print(f"  Typ danych: {args.target_col}")
    print(f"  Liczba epok: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.bs}")
    print(f"  Testowane modele: {', '.join(models_to_test)}")

    for model_type in models_to_test:
        print("\n" + "="*60)
        print(f"Trenowanie: {model_type}")
        print("="*60)

        model_args = argparse.Namespace(
            csv=args.csv,
            target_col=args.target_col,
            model=model_type,
            epochs=args.epochs,
            lr=args.lr,
            bs=args.bs,
            es=args.es,
            seed=args.seed
        )
        _, history, metrics = train(model_args)
        results[model_type] = {
            'rmse': metrics[0],
            'mae': metrics[1],
            'r2': metrics[2]
        }
        histories[model_type] = history

    return results, histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trenowanie modeli do predykcji Kp/Kpuu')

    # Argumenty dla standardowego treningu
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed dla podziału danych i reprodukowalności')
    parser.add_argument('--csv', type=str, default='data/raw/train_kpuu_log.csv',
                        help='Ścieżka do pliku CSV z danymi')
    parser.add_argument('--target_col', type=str, default='pKpuu',
                        choices=['Kpuu', 'Kp', 'pKpuu', 'pKp'],
                        help='Kolumna do przewidzenia')
    parser.add_argument('--model', type=str, default='ResidualGCN',
                        choices=['GCN', 'GIN', 'ResidualGCN', 'GraphDenseNet',
                                 'DescriptorOnly', 'Hybrid', 'All'],
                        help='Typ modelu')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maksymalna liczba epok')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--bs', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--es', type=int, default=50,
                        help='Cierpliwość early stopping')

    args = parser.parse_args()

    # Uruchom odpowiednią funkcję
    if args.model == 'All':
        train_all_models(args)
    else:
        train(args)

"""
Args:
--model GCN/GIN/ResidualGCN/GraphDenseNet/DescriptorOnly/Hybrid/All
--csv data/raw/train_kpuu.csv
--target_col pKpuu
--epochs 200
--lr (learning rate) 0.0005
--bs (batch size) 32
--es (early stopping) 50
--seed 42

train.py --model ResidualGCN --csv data/raw/train_kp_log.csv --target_col pKp --epochs 200 --lr 0.0005 --bs 32 --es 50 --seed 42
"""
