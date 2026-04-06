import pandas as pd
import numpy as np

df_kpuu = pd.read_excel(
    "Kpuu and Kp CHEMBL(1).xlsx",
    sheet_name="Kpuu 482"
)
df_kpuu = df_kpuu[df_kpuu['Standard Type'] == "K(p,uu,brain)"][[
    'Molecule ChEMBL ID', 'Smiles', 'Standard Value']]
df_kpuu.columns = ['CompoundID', 'SMILES', 'Kpuu']

df_kp = pd.read_excel(
    "Kpuu and Kp CHEMBL(1).xlsx",
    sheet_name="Kp 513"
)
df_kp = df_kp[df_kp['Standard Type'] == "Kp"][[
    'Molecule ChEMBL ID', 'Smiles', 'Standard Value']]
df_kp.columns = ['CompoundID', 'SMILES', 'Kp']

df_all = pd.merge(df_kp, df_kpuu, on=['CompoundID', 'SMILES'], how='outer')

# Sprawdzenie wartości Kpuu i Kp
print("Podstawowe statystyki:")
print(f"Liczba związków z Kp: {df_kp['CompoundID'].nunique()}")
print(f"Liczba związków z Kpuu: {df_kpuu['CompoundID'].nunique()}")
print(
    f"Liczba związków z obiema wartościami: {df_all.dropna(subset=['Kp', 'Kpuu']).shape[0]}")

print("\nZakres wartości:")
print(f"Kp - min: {df_kp['Kp'].min():.4f}, max: {df_kp['Kp'].max():.4f}")
print(
    f"Kpuu - min: {df_kpuu['Kpuu'].min():.4f}, max: {df_kpuu['Kpuu'].max():.4f}")

print("\nPrzykładowe dane:")
print(df_all.head(10))

df_all.to_csv("raw/ourdata.csv", index=False)
print(f"\nZapisano {df_all.shape[0]} związków do pliku ourdata.csv")
