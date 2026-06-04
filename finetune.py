import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_loader import create_dataloaders
from plot_utils import PlotGenerator
from train import get_model, train_epoch, evaluate, EarlyStopping

# Ustaw styl wykresów
plt.style.use('ggplot')


def freeze_layers_by_ratio(model, freeze_ratio=0.5):
    """
    Zamraża określony procent warstw modelu.

    Kolejność zamrażania: od najwcześniejszych warstw (conv1, bn1)
    do najpóźniejszych (regressor).
    """
    # Pobierz nazwy wszystkich parametrów w kolejności
    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)

    num_freeze = int(len(param_names) * freeze_ratio)

    frozen_count = 0
    for i, name in enumerate(param_names):
        param = model.get_parameter(name)
        if i < num_freeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True

    print(
        f"  Zamrożono {frozen_count} z {len(param_names)} warstw ({freeze_ratio*100:.0f}%)")
    return model


def finetune(args):

    model_name, data_type, epochs, lr, bs = args.model_path.split(
        '\\')[-2].split('_')

    folder_name = f"TL_{model_name}_{data_type}_to_{args.ft_target}_{epochs}to{args.epochs_ft}_{lr}to{args.lr_ft}_{bs}to{args.bs_ft}_{args.fr}"

    save_dir = os.path.join('results', folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dotrenowanie modelu {model_name} dla {args.ft_target}")
    print(f"{'='*60}")
    print(f"\nParametry trenowania:")
    print(f"  Plik danych: {args.ft_csv}")
    print(f"  Typ danych: {args.ft_target}")
    print(f"  Typ modelu: {model_name}")
    print(f"  Liczba epok: {args.epochs_ft}")
    print(f"  Learning rate: {args.lr_ft}")
    print(f"  Batch size: {args.bs_ft}")
    print(f"  Early stopping: {args.es_ft}")
    print(f"  Folder wyników: {save_dir}")

    use_descriptors = model_name in ['DescriptorOnly', 'Descriptor', 'descriptor',
                                     'Hybrid', 'hybrid']

    train_loader_finetune, val_loader_finetune, test_loader_finetune = create_dataloaders(
        args.ft_csv, target_col=args.ft_target, batch_size=args.bs_ft,
        use_descriptors=use_descriptors
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUżywam urządzenia: {device}")

    if use_descriptors:
        from descriptors import get_descriptor_names
        desc_dim = len(get_descriptor_names())
        model = get_model(model_name, descriptor_dim=desc_dim).to(device)
    else:
        model = get_model(model_name).to(device)

    # Załaduj pre-trenowany model
    model.load_state_dict(torch.load(args.model_path))

    # Zamroź warstwy
    if args.fr > 0:
        model = freeze_layers_by_ratio(model, args.fr)

    # Optymalizator dla trenowalnych parametrów
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.es_ft)

    print("Rozpoczynam fine-tuning...")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_r2': [],
               'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}

    for epoch in range(args.epochs_ft):
        train_loss, train_r2 = train_epoch(
            model, train_loader_finetune, optimizer, criterion, device, model_name)
        val_loss, val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader_finetune, criterion, device, model_name)

        history['train_loss'].append(train_loss)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs_ft} | "
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
    print("waluacja na zbiorze testowym")
    print("="*60)

    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_rmse, test_mae, test_r2, test_pred, test_true = evaluate(
        model, test_loader_finetune, criterion, device, model_name
    )

    print(f"\nWyniki transfer learningu:")
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
        'pretrained_model': args.model_path,
        'model': model_name,
        'target_col': args.ft_target,
        'epochs': args.epochs_ft,
        'learning_rate': args.lr_ft,
        'batch_size': args.bs_ft,
        'early_stop': args.es_ft,
        'freeze_ratio': args.fr,
        'test_loss': test_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'best_epoch': np.argmin(history['val_loss']) + 1

    }])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    plotter = PlotGenerator(save_dir=save_dir)
    plotter.plot_learning_curves(history, model_name, args.ft_target)
    plotter.plot_predictions_vs_true(
        test_true, test_pred, model_name, args.ft_target)
    plotter.plot_residuals(test_true, test_pred, model_name, args.ft_target)
    plotter.plot_errors_vs_target(
        test_true, test_pred, model_name, args.ft_target)

    print(f"\nWyniki zapisane w: {save_dir}")

    return model, history, (test_rmse, test_mae, test_r2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Transfer learning dla przewidywania Kp/Kpuu')

    parser.add_argument('--model_path', type=str, default='results\GraphDenseNet_pKpuu_E200_LR0.0005_BS32\best_model.pt',
                        help='Ścieżka do wytrenowanego modelu')

    parser.add_argument('--ft_csv', type=str, default=None,
                        help='Plik CSV(np. train_kpuu_log.csv)')

    parser.add_argument('--ft_target', type=str, default='pKpuu',
                        help='Kolumna docelowa')

    parser.add_argument('--epochs_ft', type=int, default=200,
                        help='Liczba epok')

    parser.add_argument('--lr_ft', type=float, default=0.0001,
                        help='Learning rate')

    parser.add_argument('--bs_ft', type=int,
                        default=32, help='Batch size')

    parser.add_argument('--es_ft', type=int, default=50,
                        help='Cierpliwość early stopping')

    parser.add_argument('--fr', type=float, default=0.5,
                        help='Procent warstw do zamrożenia (0-1)')

    args = parser.parse_args()

    finetune(args)

"""
Args:
--model_path: Ścieżka do wytrenowanego modelu do finetuningu (np. 'results/DescriptorOnly_Kpuu_log_E200_LR0.0005_BS32/best_model.pt')
--ft_csv: Plik CSV do fine-tuningu (np. 'data/raw/train_kpuu_log.csv')
--ft_target: Kolumna docelowa dla fine-tuningu (np. 'pKpuu')
--epochs_ft: Liczba epok fine-tuningu (np. 100)
--lr_ft: Learning rate dla fine-tuningu (np. 0.0001)
--fr: Procent warstw do zamrożenia (0-1, np. 0.5)

finetune.py --model_path results\GraphDenseNet_pKp_E200_LR0.0005_BS32\best_model.pt --ft_csv data\raw\train_kpuu_log.csv --ft_target pKpuu --epochs_ft 200 --lr_ft 0.0001 --bs_ft 32 --es_ft 50 --fr 0.5

"""
