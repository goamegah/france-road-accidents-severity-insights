# hypotheses_visuals.py

import pandas as pd
import matplotlib.pyplot as plt

# === UTILITAIRE POUR ANNOTER LES BARRES ===

def annotate_bars(ax, fontsize=12, fontweight='bold'):
    """
    Annoter les valeurs sur les barres d'un graphique.
    """
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.0f',  # Format entier
            label_type='edge',
            fontsize=fontsize,
            fontweight=fontweight,
            padding=3
        )

# === HYPOTHÈSES DOCUMENTÉES ===

def print_hypotheses():
    hypotheses = [
        "H1 : Le nombre d'accidents corporels a chuté en 2020 à cause du Covid (confinements)",
        "H2 : Les accidents sont majoritairement causés par des conducteurs masculins",
        "H3 : Les deux-roues (scooters, motos) sont surreprésentés parmi les accidents graves",
        "H4 : Les conditions météo (pluie forte, neige) augmentent la gravité des accidents",
        "H5 : Il y a plus d'accidents aux heures de pointe (8h-9h, 17h-19h)",
        "H6 : La part des accidents graves est plus élevée en Outre-Mer qu'en Métropole",
    ]
    for hyp in hypotheses:
        print("-", hyp)

# === VISUALISATIONS ===

def plot_accidents_by_year(df):
    df_year = df.groupby("an").size().reset_index(name="accident_count")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_year['an'], df_year['accident_count'])
    ax = plt.gca()
    annotate_bars(ax)
    plt.title("Nombre d'accidents par an", fontsize=14, fontweight='bold')
    plt.xlabel("Année")
    plt.ylabel("Nombre d'accidents")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_sex_vs_gravity(df):
    if 'sexe' not in df.columns or 'grav' not in df.columns:
        print("Colonnes 'sexe' ou 'grav' manquantes.")
        return
    cross_tab = pd.crosstab(df['sexe'], df['grav'])
    ax = cross_tab.plot(kind='bar', stacked=True, figsize=(10,6))
    annotate_bars(ax)
    plt.title("Sexe vs Gravité de l'accident", fontsize=14, fontweight='bold')
    plt.xlabel("Sexe (1=Homme, 2=Femme)")
    plt.ylabel("Nombre d'accidents")
    plt.legend(title="Gravité", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_vehicle_types(df):
    if 'catv_label' not in df.columns:
        print("Colonne 'catv_label' manquante. Pense à utiliser annotate_columns().")
        return
    vehicle_counts = df['catv_label'].value_counts().head(15)
    plt.figure(figsize=(12,6))
    ax = vehicle_counts.plot(kind='bar')
    annotate_bars(ax)
    plt.title("Top 15 types de véhicules impliqués dans les accidents", fontsize=14, fontweight='bold')
    plt.xlabel("Type de véhicule")
    plt.ylabel("Nombre d'accidents")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_weather_vs_gravity(df):
    if 'atm_label' not in df.columns or 'grav' not in df.columns:
        print("Colonnes 'atm_label' ou 'grav' manquantes.")
        return
    weather_gravity = pd.crosstab(df['atm_label'], df['grav'])
    ax = weather_gravity.plot(kind='bar', stacked=True, figsize=(12,6))
    annotate_bars(ax)
    plt.title("Conditions météo vs Gravité de l'accident", fontsize=14, fontweight='bold')
    plt.xlabel("Conditions météo")
    plt.ylabel("Nombre d'accidents")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_accidents_by_hour(df):
    if 'heure' not in df.columns:
        print("Colonne 'heure' manquante.")
        return
    hour_counts = df['heure'].value_counts().sort_index()
    plt.figure(figsize=(10,6))
    plt.plot(hour_counts.index, hour_counts.values, marker='o')
    plt.title("Nombre d'accidents par heure", fontsize=14, fontweight='bold')
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre d'accidents")
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.show()

def plot_zone_vs_gravity(df):
    if 'zone' not in df.columns or 'grav' not in df.columns:
        print("Colonnes 'zone' ou 'grav' manquantes.")
        return
    zone_gravity = pd.crosstab(df['zone'], df['grav'])
    ax = zone_gravity.plot(kind='bar', stacked=True, figsize=(8,6))
    annotate_bars(ax)
    plt.title("Gravité des accidents : Outre-Mer vs Métropole", fontsize=14, fontweight='bold')
    plt.xlabel("Zone")
    plt.ylabel("Nombre d'accidents")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_severity_by_year(df):
    if 'grav' not in df.columns or 'an' not in df.columns:
        print("Colonnes 'grav' ou 'an' manquantes.")
        return
    severity_mapping = {1: "Indemne", 2: "Blessé léger", 3: "Blessé hospitalisé", 4: "Tué"}
    df_filtered = df.dropna(subset=['grav', 'an']).copy()
    df_filtered['grav'] = df_filtered['grav'].map(severity_mapping)
    pivot = df_filtered.pivot_table(index='an', columns='grav', aggfunc='size', fill_value=0)
    ax = pivot.plot(kind='bar', stacked=True, figsize=(12, 7))
    annotate_bars(ax)
    plt.title('Gravité des accidents par année', fontsize=14, fontweight='bold')
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'accidents')
    plt.xticks(rotation=45)
    plt.legend(title='Gravité')
    plt.tight_layout()
    plt.show()

def plot_sex_distribution(df):
    if 'sexe' not in df.columns:
        print("Colonne 'sexe' manquante.")
        return
    sex_mapping = {1: 'Homme', 2: 'Femme'}
    df_filtered = df.dropna(subset=['sexe']).copy()
    df_filtered['sexe'] = df_filtered['sexe'].map(sex_mapping)
    counts = df_filtered['sexe'].value_counts()
    counts.plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6), startangle=90)
    plt.title('Répartition des sexes dans les accidents', fontsize=14, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

def plot_age_distribution(df):
    if 'an_nais' not in df.columns or 'an' not in df.columns:
        print("Colonnes 'an_nais' ou 'an' manquantes.")
        return
    df_filtered = df.dropna(subset=['an_nais', 'an']).copy()
    df_filtered['age'] = df_filtered['an'] - df_filtered['an_nais']
    df_filtered = df_filtered[(df_filtered['age'] >= 0) & (df_filtered['age'] <= 110)]
    bins = [0, 17, 24, 34, 44, 54, 64, 74, 84, 110]
    labels = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85+']
    df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=bins, labels=labels, right=True)
    counts = df_filtered['age_group'].value_counts().sort_index()
    ax = counts.plot(kind='bar', figsize=(10, 6))
    annotate_bars(ax)
    plt.title('Répartition par tranche d\'âge des usagers accidentés', fontsize=14, fontweight='bold')
    plt.xlabel('Tranche d\'âge')
    plt.ylabel('Nombre d\'usagers')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_weather_conditions(df, atm_mapping):
    if 'atm' not in df.columns:
        print("Colonne 'atm' manquante.")
        return
    df_filtered = df.dropna(subset=['atm']).copy()
    df_filtered['weather'] = df_filtered['atm'].map(atm_mapping)
    counts = df_filtered['weather'].value_counts().sort_values(ascending=False)
    ax = counts.plot(kind='bar', figsize=(12, 6))
    annotate_bars(ax)
    plt.title('Répartition des accidents selon les conditions météo', fontsize=14, fontweight='bold')
    plt.xlabel('Condition météo')
    plt.ylabel('Nombre d\'accidents')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_vehicle_categories(df, catv_mapping):
    if 'catv' not in df.columns:
        print("Colonne 'catv' manquante.")
        return
    df_filtered = df.dropna(subset=['catv']).copy()
    df_filtered['vehicle_category'] = df_filtered['catv'].map(catv_mapping)
    counts = df_filtered['vehicle_category'].value_counts().sort_values(ascending=False)
    ax = counts.plot(kind='bar', figsize=(14, 6))
    annotate_bars(ax)
    plt.title('Répartition des accidents par type de véhicule', fontsize=14, fontweight='bold')
    plt.xlabel('Type de véhicule')
    plt.ylabel('Nombre d\'accidents')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_accidents_by_zone(df):
    if 'zone' not in df.columns:
        print("Colonne 'zone' manquante.")
        return
    counts = df['zone'].value_counts()
    ax = counts.plot(kind='bar', figsize=(8, 5))
    annotate_bars(ax)
    plt.title('Répartition des accidents : Métropole vs Outre-Mer', fontsize=14, fontweight='bold')
    plt.xlabel('Zone')
    plt.ylabel('Nombre d\'accidents')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
