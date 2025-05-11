# data_loader.py

import pandas as pd
import os
import requests
from pathlib import Path
from fras.definitions import DATASET_DIR

def download_file(url, dest_folder):
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    local_filename = url.split("/")[-1] + ".csv"
    local_path = os.path.join(dest_folder, local_filename)
    if not os.path.exists(local_path):
        print(f"Downloading {local_filename}...")
        r = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f"Already downloaded: {local_filename}")
    return local_path

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    return df

def standardize_columns(df, year):
    if year == 2022:
        df = df.rename(columns={"accident_id": "num_acc"})
    return df

def get_zone(dep):
    try:
        dep = int(dep)
        if dep in [971, 972, 973, 974, 976]:
            return 'Outre-Mer'
        else:
            return 'Métropole'
    except (ValueError, TypeError):
        return pd.NA

def load_data(year, base_folder=DATASET_DIR):
    urls = {
        2023: {
            "carac": "https://www.data.gouv.fr/fr/datasets/r/104dbb32-704f-4e99-a71e-43563cb604f2",
            "lieux": "https://www.data.gouv.fr/fr/datasets/r/8bef19bf-a5e4-46b3-b5f9-a145da4686bc",
            "vehicules": "https://www.data.gouv.fr/fr/datasets/r/146a42f5-19f0-4b3e-a887-5cd8fbef057b",
            "usagers": "https://www.data.gouv.fr/fr/datasets/r/68848e2a-28dd-4efc-9d5f-d512f7dbe66f",
        },
        2022: {
            "carac": "https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7",
            "lieux": "https://www.data.gouv.fr/fr/datasets/r/a6ef711a-1f03-44cb-921a-0ce8ec975995",
            "vehicules": "https://www.data.gouv.fr/fr/datasets/r/c9742921-4427-41e5-81bc-f13af8bc31a0",
            "usagers": "https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8",
        },
        2021: {
            "carac": "https://www.data.gouv.fr/fr/datasets/r/85cfdc0c-23e4-4674-9bcd-79a970d7269b",
            "lieux": "https://www.data.gouv.fr/fr/datasets/r/8a4935aa-38cd-43af-bf10-0209d6d17434",
            "vehicules": "https://www.data.gouv.fr/fr/datasets/r/0bb5953a-25d8-46f8-8c25-b5c2f5ba905e",
            "usagers": "https://www.data.gouv.fr/fr/datasets/r/ba5a1956-7e82-41b7-a602-89d7dd484d7a",
        },
        2020: {
            "carac": "https://www.data.gouv.fr/fr/datasets/r/07a88205-83c1-4123-a993-cba5331e8ae0",
            "lieux": "https://www.data.gouv.fr/fr/datasets/r/e85c41f7-d4ea-4faf-877f-ab69a620ce21",
            "vehicules": "https://www.data.gouv.fr/fr/datasets/r/a66be22f-c346-49af-b196-71df24702250",
            "usagers": "https://www.data.gouv.fr/fr/datasets/r/78c45763-d170-4d51-a881-e3147802d7ee",
        },
        2019: {
            "carac": "https://www.data.gouv.fr/fr/datasets/r/e22ba475-45a3-46ac-a0f7-9ca9ed1e283a",
            "lieux": "https://www.data.gouv.fr/fr/datasets/r/2ad65965-36a1-4452-9c08-61a6c874e3e6",
            "vehicules": "https://www.data.gouv.fr/fr/datasets/r/780cd335-5048-4bd6-a841-105b44eb2667",
            "usagers": "https://www.data.gouv.fr/fr/datasets/r/36b1b7b3-84b4-4901-9163-59ae8a9e3028",
        }
    }

    if year not in urls:
        raise ValueError(f"Année {year} non disponible.")

    data_files = {}
    for key, url in urls[year].items():
        path = download_file(url, dest_folder=os.path.join(base_folder, str(year)))
        df = pd.read_csv(path, sep=';', encoding="latin1", low_memory=False)
        df = clean_column_names(df)
        df = standardize_columns(df, year)
        data_files[key] = df

    carac = data_files['carac']
    lieux = data_files['lieux']
    vehicules = data_files['vehicules']
    usagers = data_files['usagers']

    carac_lieux = carac.merge(lieux, on='num_acc', how='left')
    carac_lieux_vehicules = carac_lieux.merge(vehicules, on='num_acc', how='left')
    full_df = carac_lieux_vehicules.merge(usagers, on=['num_acc', 'id_vehicule'], how='left')

    full_df['num_acc'] = full_df['num_acc'].astype(str)

    if 'dep' in full_df.columns:
        full_df['zone'] = full_df['dep'].apply(get_zone)

    return full_df.reset_index(drop=True)

def merge_years(df_list):
    return pd.concat(df_list, ignore_index=True)

# Exemple d'utilisation :
# from data_loader import load_data, merge_years
# df_2022 = load_data(2022)
# df_2023 = load_data(2023)
# df_all = merge_years([df_2022, df_2023])
