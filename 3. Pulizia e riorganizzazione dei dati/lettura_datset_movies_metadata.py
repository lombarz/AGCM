import pandas as pd
import ast

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

df=pd.read_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata.csv",low_memory=False)
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


df_metadata_clean.to_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN.csv",index=False)