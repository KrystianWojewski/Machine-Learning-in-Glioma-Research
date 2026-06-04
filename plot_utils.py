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

    def plot_pkp_vs_pkpuu(self, df_kp_log, df_kpuu_log, save_name='pkp_vs_pkpuu'):
        """
        Rysuje zależność między pKp a pKpuu (wersje logarytmiczne).

        Args:
            df_kp_log: DataFrame z kolumnami 'SMILES' i 'pKp'
            df_kpuu_log: DataFrame z kolumnami 'SMILES' i 'pKpuu'
            save_name: nazwa pliku do zapisu
        """
        # Połącz dane na podstawie SMILES
        df_merged = pd.merge(df_kp_log, df_kpuu_log, on='SMILES', how='inner')

        if len(df_merged) == 0:
            print("Brak wspólnych związków między pKp a pKpuu")
            return None, None

        fig, ax = plt.subplots(figsize=(8, 8))

        # Punkty
        ax.scatter(df_merged['pKp'], df_merged['pKpuu'], alpha=0.6, s=50,
                   c='steelblue', edgecolors='black', linewidth=0.5)

        # Linia y = x
        min_val = min(df_merged['pKp'].min(), df_merged['pKpuu'].min())
        max_val = max(df_merged['pKp'].max(), df_merged['pKpuu'].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='y = x')

        # Linia regresji
        z = np.polyfit(df_merged['pKp'], df_merged['pKpuu'], 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), 'g-', linewidth=2,
                label=f'Regresja (y = {z[0]:.2f}x + {z[1]:.2f})')

        # Korelacja
        correlation = df_merged['pKp'].corr(df_merged['pKpuu'])

        # Ustawienia osi
        ax.set_xlabel('pKp = -log₁₀(Kp)', fontsize=12)
        ax.set_ylabel('pKpuu = -log₁₀(Kpuu)', fontsize=12)
        ax.set_title(
            f'Zależność między pKp a pKpuu\n(korelacja = {correlation:.3f})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Dodaj informację o liczbie punktów
        ax.text(0.05, 0.95, f'n = {len(df_merged)} związków', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        self._save_plot(f'{save_name}.png')

        return fig, ax

    def plot_atom_importance(self, smiles, importance, mol, save_name=None):
        """
        Rysuje wykres ważności atomów dla pojedynczej cząsteczki.

        Args:
            smiles: SMILES cząsteczki (tylko do tytułu)
            importance: lista/tablica ważności dla każdego atomu
            mol: obiekt RDKit Mol (do pobrania symboli atomów)
            save_name: nazwa pliku do zapisu
        """
        if mol is None:
            print("⚠️ Nie można wyświetlić ważności atomów – brak obiektu mol")
            return None, None

        # Przygotuj dane
        n_atoms = len(importance)
        atom_indices = list(range(n_atoms))
        atom_symbols = [mol.GetAtomWithIdx(
            i).GetSymbol() for i in range(n_atoms)]

        # Kolory: zielony = dodatnia ważność (zwiększa przenikanie),
        #        czerwony = ujemna ważność (zmniejsza przenikanie)
        colors = ['green' if imp > 0 else 'red' for imp in importance]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Wykres słupkowy
        bars = ax.bar(atom_indices, importance, color=colors,
                      alpha=0.7, edgecolor='black')

        # Dodaj etykiety symboli atomów nad słupkami
        for i, (bar, symbol, imp) in enumerate(zip(bars, atom_symbols, importance)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if imp > 0 else -0.05),
                    symbol, ha='center', va='bottom' if imp > 0 else 'top',
                    fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Indeks atomu', fontsize=12)
        ax.set_ylabel('Ważność atomu (Integrated Gradients)', fontsize=12)
        ax.set_title(
            f'Ważność atomów dla cząsteczki\n{smiles[:60]}...', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Dodaj adnotację o kolorach
        ax.text(0.98, 0.02, 'Kolor: zielony = zwiększa przenikanie\n       czerwony = zmniejsza przenikanie',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_name is None:
            save_name = f'atom_importance'
        self._save_plot(f'{save_name}.png')

        return fig, ax

    def plot_atom_importance_comparison(self, results, save_name='atom_importance_comparison'):
        """
        Rysuje porównanie ważności atomów dla wielu cząsteczek.

        Args:
            results: lista słowników z kluczami 'smiles', 'mol', 'importance', 'chembl_id', 'true_pkpuu', 'pred_pkpuu'
            save_name: nazwa pliku do zapisu
        """
        n_molecules = len(results)
        if n_molecules == 0:
            print("⚠️ Brak danych do porównania")
            return None, None

        # Wybierz maksymalną liczbę atomów dla spójnej siatki
        max_atoms = max([len(r['importance']) for r in results])

        fig, axes = plt.subplots(n_molecules, 1, figsize=(16, 3 * n_molecules))
        if n_molecules == 1:
            axes = [axes]

        for i, res in enumerate(results):
            importance = res['importance']
            mol = res['mol']
            chembl_id = res.get('chembl_id', f'Cząsteczka {i+1}')
            true_val = res.get('true_pkpuu', 0)
            pred_val = res.get('pred_pkpuu', 0)

            n_atoms = len(importance)
            atom_indices = list(range(n_atoms))
            atom_symbols = [mol.GetAtomWithIdx(
                i).GetSymbol() for i in range(n_atoms)]

            colors = ['green' if imp > 0 else 'red' for imp in importance]

            axes[i].bar(atom_indices, importance, color=colors,
                        alpha=0.7, edgecolor='black')
            axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[i].set_ylabel(
                f'{chembl_id}\nWa\u017cno\u015b\u0107', fontsize=9)
            axes[i].set_xlim(-0.5, max_atoms - 0.5)
            axes[i].set_title(f'pKpuu: true={true_val:.2f}, pred={pred_val:.2f}, błąd={abs(true_val-pred_val):.2f}',
                              fontsize=9)
            axes[i].grid(True, alpha=0.3, axis='y')

            # Dodaj etykiety atomów tylko dla co drugiego atomu (żeby nie było tłoczno)
            for j, (symbol, imp) in enumerate(zip(atom_symbols, importance)):
                if j % 2 == 0:  # co drugi atom
                    axes[i].text(j, imp + (0.02 if imp > 0 else -0.03),
                                 symbol, ha='center', va='bottom' if imp > 0 else 'top',
                                 fontsize=7)

        axes[-1].set_xlabel('Indeks atomu', fontsize=12)
        plt.suptitle('Porównanie ważności atomów dla wybranych cząsteczek',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_plot(f'{save_name}.png')

        return fig, axes

    def plot_top_atoms_summary(self, results, top_k=5, save_name='top_atoms_summary'):
        """
        Rysuje zbiorczy wykres najważniejszych atomów ze wszystkich cząsteczek.

        Args:
            results: lista słowników z kluczami 'mol', 'importance'
            top_k: liczba najważniejszych atomów do uwzględnienia z każdej cząsteczki
            save_name: nazwa pliku do zapisu
        """
        from collections import Counter

        # Zbierz wszystkie najważniejsze atomy
        atom_counter = Counter()

        for res in results:
            mol = res['mol']
            importance = res['importance']

            # Znajdź top_k najważniejszych atomów
            top_indices = np.argsort(importance)[-top_k:][::-1]

            for idx in top_indices:
                atom_symbol = mol.GetAtomWithIdx(int(idx)).GetSymbol()
                atom_counter[atom_symbol] += 1

        # Przygotuj dane do wykresu
        atom_types = list(atom_counter.keys())
        counts = list(atom_counter.values())

        # Posortuj malejąco
        sorted_idx = np.argsort(counts)[::-1]
        atom_types = [atom_types[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(atom_types, counts, color='steelblue',
                      alpha=0.7, edgecolor='black')
        ax.set_xlabel('Typ atomu', fontsize=12)
        ax.set_ylabel(
            f'Liczba wystąpień wśród {top_k} najważniejszych atomów', fontsize=12)
        ax.set_title(
            f'Najczęściej występujące ważne atomy\n(na podstawie {len(results)} cząsteczek)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Dodaj wartości na słupkach
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        self._save_plot(f'{save_name}.png')

        return fig, ax

    def plot_importance_vs_property(self, results, property_name='true_pkpuu', save_name='importance_vs_property'):
        """
        Rysuje zależność między średnią/maksymalną ważnością atomów a właściwością cząsteczki.

        Args:
            results: lista słowników z kluczami 'importance', property_name
            property_name: nazwa właściwości do porównania (np. 'true_pkpuu', 'pred_pkpuu')
            save_name: nazwa pliku do zapisu
        """
        mean_importances = [np.mean(r['importance']) for r in results]
        max_importances = [np.max(r['importance']) for r in results]
        property_values = [r[property_name] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Średnia ważność vs właściwość
        axes[0].scatter(property_values, mean_importances, alpha=0.6, s=50,
                        c='steelblue', edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(property_name, fontsize=12)
        axes[0].set_ylabel('Średnia ważność atomów', fontsize=12)
        axes[0].set_title(
            'Średnia ważność atomów a właściwość cząsteczki', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Dodaj linię trendu
        z = np.polyfit(property_values, mean_importances, 1)
        p = np.poly1d(z)
        x_line = np.array([min(property_values), max(property_values)])
        axes[0].plot(x_line, p(x_line), 'r--', linewidth=1.5,
                     label=f'Trend (r = {np.corrcoef(property_values, mean_importances)[0, 1]:.3f})')
        axes[0].legend()

        # Maksymalna ważność vs właściwość
        axes[1].scatter(property_values, max_importances, alpha=0.6, s=50,
                        c='steelblue', edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel(property_name, fontsize=12)
        axes[1].set_ylabel('Maksymalna ważność atomów', fontsize=12)
        axes[1].set_title(
            'Maksymalna ważność atomów a właściwość cząsteczki', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        z = np.polyfit(property_values, max_importances, 1)
        p = np.poly1d(z)
        axes[1].plot(x_line, p(x_line), 'r--', linewidth=1.5,
                     label=f'Trend (r = {np.corrcoef(property_values, max_importances)[0, 1]:.3f})')
        axes[1].legend()

        plt.suptitle('Zależność między ważnością atomów a wartością pKpuu',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_plot(f'{save_name}.png')

        return fig, axes


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
