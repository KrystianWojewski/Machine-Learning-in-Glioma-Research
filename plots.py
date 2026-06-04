"""
plots.py
Skrypt do generowania wykresów analitycznych dla danych i modeli.
Użycie: python plots.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import PlotGenerator


def load_data():
    """Wczytuje wszystkie potrzebne pliki danych."""
    print("Wczytywanie danych...")

    # Dane Kpuu i pKpuu
    df_kpuu = pd.read_csv('data/raw/train_kpuu.csv')
    df_pkpuu = pd.read_csv('data/raw/train_kpuu_log.csv')

    # Dane Kp i pKp
    df_kp = pd.read_csv('data/raw/train_kp.csv')
    df_pkp = pd.read_csv('data/raw/train_kp_log.csv')

    return df_kpuu, df_pkpuu, df_kp, df_pkp


if __name__ == "__main__":
    """Generuje wszystkie wykresy analityczne."""

    print("\n" + "="*60)
    print("GENEROWANIE WYKRESÓW ANALITYCZNYCH")
    print("="*60)

    plotter = PlotGenerator(save_dir='results/plots', dpi=150)

    df_kpuu, df_pkpuu, df_kp, df_pkp = load_data()

    # ========================================================================
    # WYKRES 1: Rozkład Kpuu vs pKpuu (oryginalny vs logarytmiczny)
    # ========================================================================
    print("\n1. Rozkład wartości docelowych (Kpuu vs pKpuu)...")
    plotter.plot_target_distribution(
        df_kpuu, df_pkpuu, save_name='01_target_distribution')

    # ========================================================================
    # WYKRES 2: Korelacja pKp vs pKpuu (logarytmiczne)
    # ========================================================================
    print("\n2. Korelacja pKp vs pKpuu...")
    plotter.plot_pkp_vs_pkpuu(df_pkp, df_pkpuu, save_name='02_pkp_vs_pkpuu')

    # ========================================================================
    # PODSUMOWANIE
    # ========================================================================
    print("\n" + "="*60)
    print(f"WSZYSTKIE WYKRESY ZOSTAŁY WYGENROWANE")
    print(f"Katalog: {'results/plots'}")
    print("="*60)
