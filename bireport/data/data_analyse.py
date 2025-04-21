import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === FONCTIONS UTILITAIRES ===

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

# === FONCTIONS DE BASE ===

def describe_dataset(df):
    print("=== Aperçu rapide ===")
    print(df.head())
    print("\n=== Statistiques numériques ===")
    print(df.describe(include='all'))
    print("\n=== Valeurs manquantes par colonne ===")
    print(df.isna().mean().sort_values(ascending=False))

# === ANALYSES PAR HYPOTHESE ===

def accidents_by_year(df):
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='an', data=df, palette='viridis')
    annotate_bars(ax)
    plt.title("Nombre d'accidents par année", fontsize=14, fontweight='bold')
    plt.xlabel("Année")
    plt.ylabel("Nombre d'accidents")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accidents_by_sex(df):
    if 'sexe' not in df.columns:
        print("Colonne 'sexe' absente.")
        return
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x='sexe', data=df, palette='pastel')
    annotate_bars(ax)
    plt.title("Répartition des accidents selon le sexe", fontsize=14, fontweight='bold')
    plt.xlabel("Sexe (1=Homme, 2=Femme)")
    plt.ylabel("Nombre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accidents_by_age_group(df):
    if 'an_nais' not in df.columns:
        print("Colonne 'an_nais' absente.")
        return

    df = df.copy()
    df['age'] = df['an'] - df['an_nais']

    bins = [0, 18, 30, 45, 60, 75, 100]
    labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='age_group', data=df, palette='Set2')
    annotate_bars(ax)
    plt.title("Répartition des accidents par tranches d'âge", fontsize=14, fontweight='bold')
    plt.xlabel("Tranche d'âge")
    plt.ylabel("Nombre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accidents_by_gravity(df):
    if 'grav' not in df.columns:
        print("Colonne 'grav' absente.")
        return

    plt.figure(figsize=(8,6))
    ax = sns.countplot(x='grav', data=df, palette='coolwarm')
    annotate_bars(ax)
    plt.title("Gravité des accidents", fontsize=14, fontweight='bold')
    plt.xlabel("Gravité (1=Indemne, 2=Blessé Léger, 3=Blessé Hospitalisé, 4=Tué)")
    plt.ylabel("Nombre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accidents_by_weather(df):
    if 'atm' not in df.columns:
        print("Colonne 'atm' absente.")
        return

    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='atm', data=df, palette='muted')
    annotate_bars(ax)
    plt.title("Impact de la météo sur les accidents", fontsize=14, fontweight='bold')
    plt.xlabel("Conditions météo")
    plt.ylabel("Nombre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accidents_by_vehicle_type(df):
    if 'catv' not in df.columns:
        print("Colonne 'catv' absente.")
        return

    plt.figure(figsize=(12,6))
    ax = sns.countplot(x='catv', data=df, palette='cubehelix')
    annotate_bars(ax)
    plt.title("Type de véhicule impliqué dans les accidents", fontsize=14, fontweight='bold')
    plt.xlabel("Type de véhicule")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def hypothesis_test_zone_accidents(df):
    if 'zone' not in df.columns:
        print("Colonne 'zone' absente.")
        return

    counts = df['zone'].value_counts()
    plt.figure(figsize=(6,6))
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Répartition des accidents : Métropole vs Outre-Mer', fontsize=14, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    print("\nNombre d'accidents par zone :")
    print(counts)

def accidents_by_hour(df):
    if 'heure' not in df.columns:
        print("Colonne 'heure' absente.")
        return

    plt.figure(figsize=(12,6))
    ax = sns.countplot(x='heure', data=df, palette='mako')
    annotate_bars(ax)
    plt.title("Nombre d'accidents par heure", fontsize=14, fontweight='bold')
    plt.xlabel("Heure")
    plt.ylabel("Nombre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def accidents_by_day(df):
    if 'jour' not in df.columns:
        print("Colonne 'jour' absente.")
        return

    plt.figure(figsize=(12,6))
    ax = sns.countplot(x='jour', data=df, palette='flare')
    annotate_bars(ax)
    plt.title("Nombre d'accidents par jour du mois", fontsize=14, fontweight='bold')
    plt.xlabel("Jour du mois")
    plt.ylabel("Nombre")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
