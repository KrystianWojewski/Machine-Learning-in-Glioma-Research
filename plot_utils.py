"""
plot_utils.py
Moduł z funkcjami do generowania wykresów dla projektu.
Może być używany zarówno podczas trenowania, jak i do generowania raportów.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ustaw styl wykresów

plt.style.use('ggplot')


# ============================================================================
# KLASA DO ZARZĄDZANIA WYKRESAMI
# ============================================================================


class PlotGenerator:
    """
    Klasa do generowania różnych wykresów.
    Przechowuje konfigurację i może być używana w różnych miejscach.
    """

    def __init__(self, save_dir=None, dpi=150, style=None):
        """
        Args:
            save_dir: katalog do zapisu wykresów (None = nie zapisuj)
            dpi: rozdzielczość wykresów
            style: styl matplotlib (opcjonalny)
        """
        self.save_dir = save_dir
        self.dpi = dpi
        if style and style in plt.style.available:
            plt.style.use(style)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def _save_plot(self, filename):
        """Zapisuje wykres jeśli save_dir jest podany."""
        if self.save_dir:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Zapisano: {save_path}")

    # ========================================================================
    # WYKRESY TRENINGOWE
    # ========================================================================

    def plot_learning_curves(self, history, model_name, target_col):
        """
        Rysuje krzywe uczenia się (strata i R²).

        Args:
            history: słownik z kluczami 'train_loss', 'val_loss', 'train_r2', 'val_r2'
            model_name: nazwa modelu (do tytułu)
            target_col: kolumna docelowa (do tytułu)

        Returns:
            fig, axes
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Strata (MSE)
        axes[0].plot(epochs, history['train_loss'],
                     'b-', label='Trening', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-',
                     label='Walidacja', linewidth=2)
        axes[0].set_xlabel('Epoka', fontsize=12)
        axes[0].set_ylabel('Strata (MSE)', fontsize=12)
        axes[0].set_title(
            f'{model_name}\nKrzywa uczenia się - Strata', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Najlepsza epoka
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                        label=f'Najlepszy model (epoka {best_epoch})')
        axes[0].legend(fontsize=10)

        # R²
        axes[1].plot(epochs, history['train_r2'], 'b-',
                     label='Trening', linewidth=2)
        axes[1].plot(epochs, history['val_r2'], 'r-',
                     label='Walidacja', linewidth=2)
        axes[1].set_xlabel('Epoka', fontsize=12)
        axes[1].set_ylabel('Współczynnik determinacji (R²)', fontsize=12)
        axes[1].set_title(
            f'{model_name}\nKrzywa uczenia się - R²', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Najlepsze R²
        best_val_r2 = max(history['val_r2'])
        axes[1].axhline(y=best_val_r2, color='g', linestyle='--', alpha=0.5,
                        label=f'Najlepsze R² = {best_val_r2:.4f}')
        axes[1].legend(fontsize=10)

        # plt.tight_layout()
        self._save_plot(f'{model_name}_{target_col}_learning_curves.png')

        return fig, axes

    def plot_predictions_vs_true(self, y_true, y_pred, model_name, target_col):
        """
        Rysuje wykres przewidywane vs rzeczywiste.

        Args:
            y_true: rzeczywiste wartości (lista lub numpy array)
            y_pred: przewidywane wartości (lista lub numpy array)
            model_name: nazwa modelu
            target_col: kolumna docelowa

        Returns:
            fig, ax
        """
        # Konwersja na numpy array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Punkty
        ax.scatter(y_true, y_pred, alpha=0.5, c='steelblue', s=50,
                   edgecolors='black', linewidth=0.5)

        # Linia idealnej predykcji
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Idealna predykcja (y = x)')

        # Linia regresji
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2,
                label=f'Linia regresji (y = {z[0]:.2f}x + {z[1]:.2f})')

        # Metryki
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))

        ax.set_xlabel(f'Rzeczywiste {target_col}', fontsize=12)
        ax.set_ylabel(f'Przewidywane {target_col}', fontsize=12)
        ax.set_title(
            f'{model_name}\nPrzewidywane vs rzeczywiste wartości', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Metryki na wykresie
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # plt.tight_layout()
        self._save_plot(f'{model_name}_{target_col}_predictions.png')

        return fig, ax

    def plot_residuals(self, y_true, y_pred, model_name, target_col):
        """
        Rysuje wykres reszt i QQ-plot.

        Args:
            y_true: rzeczywiste wartości (lista lub numpy array)
            y_pred: przewidywane wartości (lista lub numpy array)
            model_name: nazwa modelu
            target_col: kolumna docelowa

        Returns:
            fig, axes
        """
        # Konwersja na numpy array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Wykres reszt vs przewidywane
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, c='steelblue',
                        edgecolors='black', linewidth=0.5)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel(f'Przewidywane {target_col}', fontsize=12)
        axes[0].set_ylabel('Reszta (rzeczywiste - przewidywane)', fontsize=12)
        axes[0].set_title(f'{model_name}\nWykres reszt', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # QQ-plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(
            'Wykres kwantyl-kwantyl (QQ-plot)\nsprawdzenie normalności reszt', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Statystyki
        axes[0].text(0.05, 0.95, f'Średnia reszt: {np.mean(residuals):.4f}\nOdch. std.: {np.std(residuals):.4f}',
                     transform=axes[0].transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # plt.tight_layout()
        self._save_plot(f'{model_name}_{target_col}_residuals.png')

        return fig, axes

    def plot_errors_vs_target(self, y_true, y_pred, model_name, target_col):
        """
        Rysuje wykres błędów w funkcji wartości docelowej.

        Args:
            y_true: rzeczywiste wartości (lista lub numpy array)
            y_pred: przewidywane wartości (lista lub numpy array)
            model_name: nazwa modelu
            target_col: kolumna docelowa

        Returns:
            fig, ax
        """
        # Konwersja na numpy array
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = np.abs(y_true - y_pred)

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(y_true, errors, c=errors, cmap='RdYlGn_r',
                             s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(f'Rzeczywiste {target_col}', fontsize=12)
        ax.set_ylabel('Błąd bezwzględny |true - pred|', fontsize=12)
        ax.set_title(
            f'{model_name}\nBłędy predykcji w funkcji wartości docelowej', fontsize=12)
        ax.axhline(y=np.mean(errors), color='red', linestyle='--',
                   label=f'Średni błąd: {np.mean(errors):.3f}')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Błąd bezwzględny', fontsize=10)

        # plt.tight_layout()
        self._save_plot(f'{model_name}_{target_col}_errors.png')

        return fig, ax

    # ========================================================================
    # WYKRESY ANALIZY DANYCH
    # ========================================================================

    def plot_target_distribution(self, df_kpuu, df_pkpuu, save_name='target_distribution'):
        """
        Rysuje rozkład wartości docelowych (Kpuu vs pKpuu).

        Args:
            df_kpuu: DataFrame z kolumną 'Kpuu'
            df_pkpuu: DataFrame z kolumną 'pKpuu'
            save_name: nazwa pliku do zapisu
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Kpuu
        kpuu_data = df_kpuu['Kpuu'].dropna()
        axes[0].hist(kpuu_data, bins=50, color='steelblue',
                     edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Kpuu', fontsize=12)
        axes[0].set_ylabel('Liczba cząsteczek', fontsize=12)
        axes[0].set_title('Rozkład wartości Kpuu', fontsize=12)
        axes[0].axvline(kpuu_data.median(), color='red', linestyle='--',
                        label=f'Mediana: {kpuu_data.median():.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # pKpuu
        pkpuu_data = df_pkpuu['pKpuu'].dropna()
        axes[1].hist(pkpuu_data, bins=50, color='steelblue',
                     edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('pKpuu = -log₁₀(Kpuu)', fontsize=12)
        axes[1].set_ylabel('Liczba cząsteczek', fontsize=12)
        axes[1].set_title(
            'Rozkład wartości pKpuu (logarytmiczny)', fontsize=12)
        axes[1].axvline(pkpuu_data.median(), color='red', linestyle='--',
                        label=f'Mediana: {pkpuu_data.median():.4f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Porównanie rozkładów Kpuu i pKpuu',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_plot(f'{save_name}.png')

        return fig, axes

    def plot_kp_vs_kpuu(self, df_kp, df_kpuu, save_name='kp_vs_kpuu'):
        """
        Rysuje zależność między Kp a Kpuu.

        Args:
            df_kp: DataFrame z kolumnami 'SMILES' i 'Kp'
            df_kpuu: DataFrame z kolumnami 'SMILES' i 'Kpuu'
            save_name: nazwa pliku do zapisu
        """
        df_merged = pd.merge(df_kp, df_kpuu, on='SMILES', how='inner')

        if len(df_merged) == 0:
            print("Brak wspólnych związków między Kp a Kpuu")
            return None, None

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(df_merged['Kp'], df_merged['Kpuu'], alpha=0.6, s=50,
                   c='steelblue', edgecolors='black', linewidth=0.5)

        min_val = min(df_merged['Kp'].min(), df_merged['Kpuu'].min())
        max_val = max(df_merged['Kp'].max(), df_merged['Kpuu'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='y = x')

        z = np.polyfit(df_merged['Kp'], df_merged['Kpuu'], 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2,
                label=f'Regresja (y = {z[0]:.2f}x + {z[1]:.2f})')

        correlation = df_merged['Kp'].corr(df_merged['Kpuu'])

        ax.set_xlabel('Kp (całkowity współczynnik)', fontsize=12)
        ax.set_ylabel('Kpuu (współczynnik dla frakcji wolnej)', fontsize=12)
        ax.set_title(
            f'Zależność między Kp a Kpuu\n(korelacja = {correlation:.3f})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'n = {len(df_merged)} związków', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self._save_plot(f'{save_name}.png')

        return fig, ax


# ============================================================================
# FUNKCJE POMOCNICZE (do użytku bez instancji klasy)
# ============================================================================

def quick_plot_learning_curves(history, model_name, target_col, save_path=None):
    """Szybka funkcja do wygenerowania krzywych uczenia się."""
    plotter = PlotGenerator(save_dir=os.path.dirname(
        save_path) if save_path else None)
    return plotter.plot_learning_curves(history, model_name, target_col)


def quick_plot_predictions(y_true, y_pred, model_name, target_col, save_path=None):
    """Szybka funkcja do wygenerowania wykresu przewidywań."""
    plotter = PlotGenerator(save_dir=os.path.dirname(
        save_path) if save_path else None)
    return plotter.plot_predictions_vs_true(y_true, y_pred, model_name, target_col)
