import pandas as pd

df=pd.read_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN_ID.csv',low_memory=False)
print(df.head())
print(df.info())  # Vedi il tipo di dati
print(df.describe())  # Statistiche di base

#rimuovo titoli con numeri
# Filtra i titoli che contengono solo caratteri alfabetici, spazi e segni di punteggiatura
df_cleaned = df[df['title'].str.contains(r'^[^\d]+$', regex=True, na=False)]

# Converti 'budget' e 'revenue' in valori numerici (float), rimuovendo eventuali errori
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

# Rimuovi righe con budget pari a 0
df_cleaned = df[df['budget'] != 0]

# Rimuovi righe con revenue pari a 0 (su df_cleaned, non su df originale)
df_cleaned = df_cleaned[df_cleaned['revenue'] != 0]




# Salva il file CSV pulito
df_cleaned.to_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN_ID_budget.csv', index=False)

print(df_cleaned.head())
print(df_cleaned.info())  # Vedi il tipo di dati
print(df_cleaned.describe())  # Statistiche di base
