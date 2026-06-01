# prepare_to_train.py
import pandas as pd
import numpy as np
from rdkit import Chem

# Wczytaj dane
df = pd.read_csv("data/raw/ourdata.csv")

print("=== PRZYGOTOWANIE DANYCH DLA MODELU ===")

# 1. Sprawdź poprawność SMILES


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


df['valid_smiles'] = df['SMILES'].apply(is_valid_smiles)
print(f"Poprawne SMILES: {df['valid_smiles'].sum()}/{len(df)}")

# 2. Usuń niepoprawne SMILES
df = df[df['valid_smiles']].drop('valid_smiles', axis=1)

# ========================================================================
# KROK KLUCZOWY: UŚREDNIENIE DLA TEGO SAMEGO ZWIĄZKU
# ========================================================================
print("\n=== UŚREDNIANIE DUPLIKATÓW ===")

# Dla Kpuu
df_kpuu_raw = df.dropna(subset=['Kpuu']).copy()
print(
    f"Przed uśrednieniem - Kpuu: {len(df_kpuu_raw)} wierszy, {df_kpuu_raw['CompoundID'].nunique()} unikalnych związków")

df_kpuu = df_kpuu_raw.groupby(['CompoundID', 'SMILES']).agg({
    'Kpuu': 'mean'  # Użyj średniej zamiast mediany (ale można też 'median')
}).reset_index()
print(f"Po uśrednieniu - Kpuu: {len(df_kpuu)} wierszy")

# Dla Kp
df_kp_raw = df.dropna(subset=['Kp']).copy()
print(
    f"Przed uśrednieniem - Kp: {len(df_kp_raw)} wierszy, {df_kp_raw['CompoundID'].nunique()} unikalnych związków")

df_kp = df_kp_raw.groupby(['CompoundID', 'SMILES']).agg({
    'Kp': 'mean'
}).reset_index()
print(f"Po uśrednieniu - Kp: {len(df_kp)} wierszy")

# Związki z obiema wartościami (po uśrednieniu)
df_both = pd.merge(df_kp, df_kpuu, on=['CompoundID', 'SMILES'], how='inner')
print(f"Związki z obiema wartościami (po uśrednieniu): {len(df_both)}")

# 3. Korelacja między Kp a Kpuu (po uśrednieniu)
if len(df_both) > 0:
    correlation = df_both['Kp'].corr(df_both['Kpuu'])
    print(f"\nKorelacja między Kp a Kpuu (po uśrednieniu): {correlation:.3f}")

# 4. Zapisz dane treningowe
df_kpuu[['CompoundID', 'SMILES', 'Kpuu']].to_csv(
    "data/raw/train_kpuu.csv", index=False)
print(f"\n✅ Zapisano {len(df_kpuu)} związków do 'train_kpuu.csv'")

df_kp[['CompoundID', 'SMILES', 'Kp']].to_csv(
    "data/raw/train_kp.csv", index=False)
print(f"✅ Zapisano {len(df_kp)} związków do 'train_kp.csv'")

# 5. Transformacja logarytmiczna
df_kpuu['pKpuu'] = -np.log10(df_kpuu['Kpuu'])
df_kp['pKp'] = -np.log10(df_kp['Kp'])

print(f"\n=== PO TRANSFORMACJI LOGARYTMICZNEJ (po uśrednieniu) ===")
print(
    f"pKpuu - min: {df_kpuu['pKpuu'].min():.3f}, max: {df_kpuu['pKpuu'].max():.3f}")
print(f"pKp - min: {df_kp['pKp'].min():.3f}, max: {df_kp['pKp'].max():.3f}")

# Zapisz wersję logarytmiczną
df_kpuu[['CompoundID', 'SMILES', 'Kpuu', 'pKpuu']].to_csv(
    "data/raw/train_kpuu_log.csv", index=False)
df_kp[['CompoundID', 'SMILES', 'Kp', 'pKp']].to_csv(
    "data/raw/train_kp_log.csv", index=False)

print("\n✅ Zapisano wersje z transformacją logarytmiczną")

# 6. Podsumowanie
print("\n" + "="*50)
print("PODSUMOWANIE PO UŚREDNIENIU")
print("="*50)
print(f"Kpuu: {len(df_kpuu)} unikalnych związków")
print(f"Kp: {len(df_kp)} unikalnych związków")
print(f"Oba wskaźniki: {len(df_both)} związków")
print(f"Korelacja Kp vs Kpuu: {correlation:.3f}")
