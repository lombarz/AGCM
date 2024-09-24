import pandas as pd
import ast

def contiene_numeri(s):
    return any(char.isdigit() for char in s)
# Carica il dataset
df = pd.read_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN_ID.csv', low_memory=False)
# Funzione per controllare se una stringa contiene numeri
# Mostra informazioni di base sul dataset
print(df.head())
print(df.info())  # Vedi il tipo di dati
print(df.describe())  # Statistiche di base

# Se la colonna 'production_company' è in formato stringa di lista, convertila in lista Python
df['production_companies_list'] = df['production_companies_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Rimuovi le case di produzione con numeri
df['production_companies_list'] = df['production_companies_list'].apply(lambda companies: [company for company in companies if not contiene_numeri(company)])

# Rimuovi le righe dove non è rimasta alcuna casa di produzione
df_cleaned = df[df['production_companies_list'].apply(lambda x: len(x) > 0)]


# Filtra i titoli che NON contengono numeri (rimuovi titoli con numeri come 'se7en')
df_cleaned = df[~df['title'].str.contains(r'\d', regex=True, na=False)]

# Converti 'budget' e 'revenue' in valori numerici (float), rimuovendo eventuali errori
df_cleaned['budget'] = pd.to_numeric(df_cleaned['budget'], errors='coerce')
df_cleaned['revenue'] = pd.to_numeric(df_cleaned['revenue'], errors='coerce')

# Rimuovi righe con budget pari a 0
df_cleaned = df_cleaned[df_cleaned['budget'] != 0]

# Rimuovi righe con revenue pari a 0
df_cleaned = df_cleaned[df_cleaned['revenue'] != 0]

# Salva il file CSV pulito
df_cleaned.to_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN_ID_budget.csv', index=False)

# Mostra il risultato del dataframe pulito
print(df_cleaned.head())
print(df_cleaned.info())  # Vedi il tipo di dati
print(df_cleaned.describe())  # Statistiche di base

