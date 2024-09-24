import pandas as pd
import ast


#LETTURA DATASET
def converti_in_lista(stringa):
    try:
        return ast.literal_eval(stringa)  # Converte la stringa in una lista/dizionario
    except:
        return None  # Se c'è un errore, restituisce None

def controllo_lista(lista):
    if isinstance(lista, list):  # Controlla se è una lista
        return all(isinstance(d, dict) for d in lista)  # Verifica se tutti gli elementi della lista sono dizionari
    return False
def aggiusto_liste(data,name_column,name_value,name_new_column):
    data[name_column] = data[name_column].apply(converti_in_lista)#converte in lista a meno che non ci siano errori, i quel caso ritorna un None
    data = data[data[name_column].apply(controllo_lista)]#se è none, elimina la riga
    data[name_new_column] = data[name_column].apply(lambda x: [d[name_value] for d in x if name_value in d])
    data.drop(columns=[name_column],inplace=True)#alla fine elimino la colonna, non mi serve
    return data

df=pd.read_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata.csv",low_memory=False)
df_metadata_clean=df
#togliamo eventuali duplicati e copie inutili
df_metadata_clean.drop_duplicates(inplace=True)
df_metadata_clean.drop(columns=['belongs_to_collection','homepage','imdb_id','original_title','overview','poster_path','status','tagline','video'],inplace=True)
#aggiustiamo le colonne genres, production_companies, production_countries e spoken_lenguages, creando nuove colonne
df_metadata_clean=aggiusto_liste(df_metadata_clean,'genres','name','genres_list')
df_metadata_clean=aggiusto_liste(df_metadata_clean,'production_companies','name','production_companies_list')
df_metadata_clean=aggiusto_liste(df_metadata_clean,'production_countries','name','production_countries_list')
df_metadata_clean=aggiusto_liste(df_metadata_clean,'spoken_languages','name','spoken_languages_list')
#lasciare solo l'anno di uscita, rimuovendo la data
df_metadata_clean['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')#alcune date non erano scritte nel giusto formato
df_metadata_clean['release_year'] = df_metadata_clean['release_date'].dt.year
df_metadata_clean['release_year'] = df_metadata_clean['release_year'].fillna(0).astype(int)# Riempire i valori NaN con un valore predefinito (es. 0) e convertire in intero
df_metadata_clean.drop(columns=['release_date'],inplace=True)

df_metadata_clean.to_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean.csv",index=False)





#AGGIUSTO ID
df1=pd.read_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean.csv",low_memory=False)
df2=pd.read_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/links.csv",low_memory=False)
df1 = df1.drop_duplicates(subset='id')
df2 = df2.drop_duplicates(subset='movieId')
df2 = df2.drop_duplicates(subset='tmdbId')
mappa_id = df2.set_index('tmdbId')['movieId']

# Creiamo una nuova colonna con il nuovo ID usando .map()
df1['movieId'] = df1['id'].map(mappa_id)
df1 = df1.drop(columns=['id'])
df1.to_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean_id.csv",index=False)




#RIMOZIONE ZERO BUDGET
def contiene_numeri(s):
    return any(char.isdigit() for char in s)
# Carica il dataset
df = pd.read_csv('/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean_id.csv', low_memory=False)
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
df_cleaned.to_csv('/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean_id_budget.csv', index=False)
print(df_cleaned.head())
print(df_cleaned.info())  # Vedi il tipo di dati
print(df_cleaned.describe())  # Statistiche di base


#RIMOZIONE COUNT<MEDIA
df = pd.read_csv('/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean_id_budget.csv')

percentile_50 = df['vote_count'].quantile(0.50)

# Filtra il dataset per mantenere solo le righe dove 'vote_count' è >= al 25° percentile
df_filtered = df[df['vote_count'] >= percentile_50]

# Salva il dataset pulito (opzionale)
df_filtered.to_csv('/Users/giacomovecchio/Desktop/corso crif/progetto/codici finali/movies_metadata_clean_id_budget_rec50.csv', index=False)
print(df_filtered)
print(df_filtered.head())
print(df_filtered.info())
print(df_filtered.describe())



