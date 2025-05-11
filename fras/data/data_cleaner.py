# data_cleaning.py

import pandas as pd

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes : minuscule, snake_case.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    return df

def clean_dataset(df: pd.DataFrame, remove_dups: bool = True, dup_subset=None, dup_keep='first') -> pd.DataFrame:
    """
    Nettoie un DataFrame :
    - Suppression caractères parasites
    - Harmonisation NA
    - Correction des heures/minutes
    - Typage compatible avec NA
    """
    # Nettoyage texte (suppression \xa0 et espaces)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.replace('\xa0', '', regex=False).str.strip()

    # Convertir pour accepter les NA
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype('Int64')  # Integer nullable
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('Float64')  # Float nullable

    # Remplacement des codes -1, 0, etc. par NA
    to_replace = [-1, -1.0, 0, 0.0, "-1", "0", " -1", " 0"]
    df = df.replace(to_replace, pd.NA)

    # Correction heure/minute si hrmn existe
    if 'hrmn' in df.columns:
        df['hrmn'] = df['hrmn'].astype(str).str.replace(':', '', regex=False).str.zfill(4)
        df['heure'] = pd.to_numeric(df['hrmn'].str[:2], errors='coerce')
        df['minute'] = pd.to_numeric(df['hrmn'].str[2:], errors='coerce')
    if remove_dups:
        df = df.drop_duplicates(subset=dup_subset, keep=dup_keep)

    return df.reset_index(drop=True)

# Exemple d'utilisation :
# from data_cleaning import clean_dataset
# df_2023_clean = clean_dataset(df_2023)
