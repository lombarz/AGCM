import pandas as pd


df = pd.read_csv('PROGETTO\movies_metadata_CLEAN_ID_budget.csv')

# Calcola il 25° percentile della colonna 'vote_count'
percentile_50 = df['vote_count'].quantile(0.50)

# Filtra il dataset per mantenere solo le righe dove 'vote_count' è >= al 25° percentile
df_filtered = df[df['vote_count'] >= percentile_50]

# Salva il dataset pulito (opzionale)
df_filtered.to_csv('PROGETTO\movies_metadata_50_perc.csv', index=False)

# Mostra il risultato
print(df_filtered)
print(df_filtered.head())
print(df_filtered.info())
print(df_filtered.describe())