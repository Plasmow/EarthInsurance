import pandas as pd

# Charger les données
path = r"Tornadotracks\Tornado_Tracks_1950_2017_1_7964592706304725094.csv"
df_tornado = pd.read_csv(path)

# Convertir Date en datetime
df_tornado['Date'] = pd.to_datetime(df_tornado['Date'])

# Filtrer entre 2017 et 2024
df_2017_2024 = df_tornado[
    (df_tornado['Date'].dt.year >= 2017) & 
    (df_tornado['Date'].dt.year <= 2024)
].copy()

# Sélectionner colonnes pertinentes
df_2017_2024 = df_2017_2024[[
    "Date",
    "Year",
    "Month",
    "State Abbreviation",
    "EF Scale (unaltered or previous rating)",
    "Magnitude",
    "Starting Latitude",
    "Starting Longitude",
    "Ending Latitude",
    "Ending Longitude",
    "Fatalities",
    "Injuries",
    "Property Loss"
]].copy()

# Renommer
df_2017_2024.columns = [
    "Date", "Year", "Month", "State", "EF_Scale", 
    "Magnitude", "Start_Lat", "Start_Lon", "End_Lat", "End_Lon",
    "Fatalities", "Injuries", "Property_Loss"
]

print(f"Total tornades (2017-2024): {len(df_2017_2024)}")
print(f"\nPar année :")
print(df_2017_2024['Year'].value_counts().sort_index())



