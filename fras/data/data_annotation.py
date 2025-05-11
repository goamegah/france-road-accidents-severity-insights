# data_annotation.py

# Mappings des colonnes codées

ATM_MAPPING = {
    -1: "Non renseigné",
    1: "Normale",
    2: "Pluie légère",
    3: "Pluie forte",
    4: "Neige / grêle",
    5: "Brouillard / fumée",
    6: "Vent fort / tempête",
    7: "Temps éblouissant",
    8: "Temps couvert",
    9: "Autre",
}

CATV_MAPPING = {
    0: "Indéterminable",
    1: "Bicyclette",
    2: "Cyclomoteur <50cm3",
    3: "Voiturette",
    7: "Voiture légère (VL)",
    10: "Véhicule utilitaire léger",
    30: "Scooter <50cm3",
    31: "Moto 50-125cm3",
    32: "Scooter 50-125cm3",
    33: "Moto >125cm3",
    34: "Scooter >125cm3",
    35: "Quad léger",
    36: "Quad lourd",
    37: "Autobus",
    38: "Autocar",
    50: "EDP à moteur",
    60: "EDP sans moteur",
    80: "VAE",
    99: "Autre véhicule",
}


# Fonction d'annotation
def annotate_columns(df):
    """
    Ajoute des colonnes annotées lisibles dans le DataFrame.
    """
    df = df.copy()
    if 'atm' in df.columns:
        df['atm_label'] = df['atm'].map(ATM_MAPPING)
    if 'catv' in df.columns:
        df['catv_label'] = df['catv'].map(CATV_MAPPING)
    return df
