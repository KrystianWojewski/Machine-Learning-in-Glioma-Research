import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Wczytaj oczyszczone dane (bez usuwania outlierów)
df = pd.read_csv("data/brain_to_plasma/raw/ourdata.csv")

print("=== PRZYGOTOWANIE DANYCH DLA MODELU ===")

# 1. Sprawdźmy poprawność SMILES


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

# 3. Wybierzmy, co chcemy przewidywać
# Możemy zrobić osobne modele dla Kp i Kpuu, albo jeden model dla obu

# Opcja 1: Tylko związki z Kpuu (to jest bardziej interesujące dla BBB)
df_kpuu = df.dropna(subset=['Kpuu']).copy()
print(f"\nZwiązki z Kpuu: {len(df_kpuu)}")

# Opcja 2: Tylko związki z Kp
df_kp = df.dropna(subset=['Kp']).copy()
print(f"Związki z Kp: {len(df_kp)}")

# Opcja 3: Tylko związki z obiema wartościami (do testowania korelacji)
df_both = df.dropna(subset=['Kp', 'Kpuu']).copy()
print(f"Związki z obiema wartościami: {len(df_both)}")

# 4. Sprawdźmy korelację między Kp i Kpuu
if len(df_both) > 0:
    correlation = df_both['Kp'].corr(df_both['Kpuu'])
    print(f"\nKorelacja między Kp a Kpuu: {correlation:.3f}")

# 5. Przygotujmy dane dla modelu - zapiszemy w formacie CSV z SMILES i wartością
# To będzie nasz plik treningowy

# Dla Kpuu (ważniejsze dla glejaka - przenikanie przez BBB)
df_kpuu[['CompoundID', 'SMILES', 'Kpuu']].to_csv(
    "raw/train_kpuu.csv", index=False)
print(f"\nZapisano {len(df_kpuu)} związków do 'train_kpuu.csv'")

# Dla Kp
df_kp[['CompoundID', 'SMILES', 'Kp']].to_csv(
    "raw/train_kp.csv", index=False)
print(f"Zapisano {len(df_kp)} związków do 'train_kp.csv'")

# 6. Pokażmy przykładowe dane
print("\n=== PRZYKŁADOWE DANE DLA Kpuu ===")
print(df_kpuu[['CompoundID', 'SMILES', 'Kpuu']].head(10))

# 7. Ważne: wartości Kpuu są w różnych jednostkach?
# Sprawdźmy unikalne wartości
print("\n=== UNIKALNE WARTOŚCI Kpuu (pierwsze 10) ===")
print(sorted(df_kpuu['Kpuu'].unique())[:10])
print("...")
print(sorted(df_kpuu['Kpuu'].unique())[-10:])

# 8. Transformacja logarytmiczna (często używana w chemii medycznej)
df_kpuu['pKpuu'] = -np.log10(df_kpuu['Kpuu'])
df_kp['pKp'] = -np.log10(df_kp['Kp'])

print("\n=== PO TRANSFORMACJI LOGARYTMICZNEJ ===")
print(
    f"pKpuu - min: {df_kpuu['pKpuu'].min():.3f}, max: {df_kpuu['pKpuu'].max():.3f}")
print(f"pKp - min: {df_kp['pKp'].min():.3f}, max: {df_kp['pKp'].max():.3f}")

# Zapisz również wersję z transformacją
df_kpuu[['CompoundID', 'SMILES', 'Kpuu', 'pKpuu']].to_csv(
    "raw/train_kpuu_log.csv", index=False)
df_kp[['CompoundID', 'SMILES', 'Kp', 'pKp']].to_csv(
    "raw/train_kp_log.csv", index=False)
print("\nZapisano również wersje z transformacją logarytmiczną")
