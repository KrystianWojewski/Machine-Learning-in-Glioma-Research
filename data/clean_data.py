import pandas as pd
import numpy as np

# Wczytaj dane
df = pd.read_csv("raw/ourdata.csv")

print("=== PRZED CZYSZCZENIEM ===")
print(f"Liczba wszystkich wierszy: {len(df)}")
print(f"Wiersze z brakującym Kp: {df['Kp'].isna().sum()}")
print(f"Wiersze z brakującym Kpuu: {df['Kpuu'].isna().sum()}")

# 1. Usuń wiersze, gdzie brakuje BOTH wartości (nie mają ani Kp ani Kpuu)
df_clean = df.dropna(subset=['Kp', 'Kpuu'], how='all')

# 2. Dla każdego związku, weźmy średnią z dostępnych pomiarów
# To rozsądne podejście, ponieważ różne wartości dla tego samego związku
# mogą wynikać z różnych warunków eksperymentalnych
df_grouped = df_clean.groupby(['CompoundID', 'SMILES']).agg({
    'Kp': 'mean',
    'Kpuu': 'mean'
}).reset_index()

print("\n=== PO CZYSZCZENIU ===")
print(f"Liczba unikalnych związków: {len(df_grouped)}")
print(f"Liczba związków z Kp: {df_grouped['Kp'].notna().sum()}")
print(f"Liczba związków z Kpuu: {df_grouped['Kpuu'].notna().sum()}")
print(
    f"Liczba związków z obiema wartościami: {df_grouped.dropna(subset=['Kp', 'Kpuu']).shape[0]}")

# 3. Sprawdźmy zakres wartości po uśrednieniu
print("\n=== ZAKRES WARTOŚCI PO UŚREDNIENIU ===")
print(
    f"Kp - min: {df_grouped['Kp'].min():.4f}, max: {df_grouped['Kp'].max():.4f}")
print(
    f"Kpuu - min: {df_grouped['Kpuu'].min():.4f}, max: {df_grouped['Kpuu'].max():.4f}")

# 4. Zobaczmy jak wyglądają dane dla przykładowego związku
example = df_grouped[df_grouped['CompoundID'] == 'CHEMBL1112']
print("\n=== DLA PRZYKŁADOWEGO ZWIĄZKU (CHEMBL1112) ===")
print(example)

# 5. Zobaczmy rozkład wartości (ważne dla modelu regresyjnego)
print("\n=== ROZKŁAD WARTOŚCI ===")
print("Kp - percentyle:")
print(df_grouped['Kp'].describe(percentiles=[.25, .5, .75, .9, .95]))
print("\nKpuu - percentyle:")
print(df_grouped['Kpuu'].describe(percentiles=[.25, .5, .75, .9, .95]))

# 6. Zapisujemy oczyszczone dane
df_grouped.to_csv(
    "raw/ourdata_clean_unique.csv", index=False)
print("\n Zapisano oczyszczone dane do 'ourdata_clean_unique.csv'")

# 7. Dodatkowo - usuńmy wartości odstające (outliers) jeśli są bardzo ekstremalne
# To może pomóc w trenowaniu modelu
Q1_Kp = df_grouped['Kp'].quantile(0.25)
Q3_Kp = df_grouped['Kp'].quantile(0.75)
IQR_Kp = Q3_Kp - Q1_Kp
lower_bound_Kp = Q1_Kp - 1.5 * IQR_Kp
upper_bound_Kp = Q3_Kp + 1.5 * IQR_Kp

Q1_Kpuu = df_grouped['Kpuu'].quantile(0.25)
Q3_Kpuu = df_grouped['Kpuu'].quantile(0.75)
IQR_Kpuu = Q3_Kpuu - Q1_Kpuu
lower_bound_Kpuu = Q1_Kpuu - 1.5 * IQR_Kpuu
upper_bound_Kpuu = Q3_Kpuu + 1.5 * IQR_Kpuu

print("\n=== WYKRYWANIE WARTOŚCI ODSTAJĄCYCH ===")
print(
    f"Kp - dolna granica: {lower_bound_Kp:.4f}, górna granica: {upper_bound_Kp:.4f}")
print(
    f"Kpuu - dolna granica: {lower_bound_Kpuu:.4f}, górna granica: {upper_bound_Kpuu:.4f}")

# Opcjonalnie - możemy stworzyć wersję bez wartości odstających
df_no_outliers = df_grouped[
    (df_grouped['Kp'].between(lower_bound_Kp, upper_bound_Kp) | df_grouped['Kp'].isna()) &
    (df_grouped['Kpuu'].between(lower_bound_Kpuu,
     upper_bound_Kpuu) | df_grouped['Kpuu'].isna())
]
print(
    f"\nPo usunięciu wartości odstających zostało {len(df_no_outliers)} związków")
df_no_outliers.to_csv(
    "raw/ourdata_cleaned_no_outliers.csv", index=False)
