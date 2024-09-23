import pandas as pd

df=pd.read_csv("C://Users//Loredana//Desktop//Academy MM//PYTHON//PROGETTO//movies_metadata_CLEAN_ID.csv",low_memory=False)
print(df.head())
print(df.info())  # Vedi il tipo di dati
print(df.describe())  # Statistiche di base


#rimuovo osservazioni con budget 0
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df_cleaned = df[df['budget'] != 0]

#rimuovo osservazioni con revenue 0
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df_cleaned = df[df['revenue'] != 0]

#rimuovo titoli con numeri
titoli_giu = df['title'].str.isalpha()
df_cleaned = df[titoli_giu]


# Salva il file CSV pulito
df_cleaned.to_csv('C://Users//Loredana//Desktop//Academy MM//PYTHON//PROGETTO//movies_metadata_CLEAN_id_budget.csv', index=False)

print(df_cleaned.head())
print(df_cleaned.info())  # Vedi il tipo di dati
print(df_cleaned.describe())  # Statistiche di base
