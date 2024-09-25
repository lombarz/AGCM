#Caricamento dei Dataset: Assicurati di aver caricato entrambi i file CSV correttamente.

import pandas as pd

# Carica i file CSV
movies_df = pd.read_csv('/Users/cosimozaccaria/Desktop/corso crif/PROGETTO/movies_metadata_clean_id_budget_rec50.csv')
ratings_df = pd.read_csv('/Users/cosimozaccaria/Desktop/corso crif/PROGETTO/The Movies Dataset/ratings_small.csv')

#Unione dei Dataset: Unisci i due dataset in base alla colonna movieId, che è presente in entrambi i file.
# Unisci i dataset sui film e sulle recensioni
df_merged = pd.merge(movies_df, ratings_df, on='movieId')

#Verifica dell'Unione: Controlla le prime righe del dataframe risultante per assicurarti che l'unione sia corretta.
print(df_merged.head())

#Normalizzazione delle Colonne Numeriche: Ora puoi normalizzare le colonne numeriche come previsto.
from sklearn.preprocessing import MinMaxScaler

# Seleziona le colonne numeriche da normalizzare
numeric_columns = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

# Inizializza lo scaler
scaler = MinMaxScaler()

# Normalizza le colonne numeriche
df_merged[numeric_columns] = scaler.fit_transform(df_merged[numeric_columns])

# Visualizza le prime righe dei dati normalizzati
print(df_merged[numeric_columns].head())

#Ottimo! Proseguiamo con la preparazione e la pulizia dei dati. Nel passaggio precedente, abbiamo caricato i dati 
# e identificato le colonne numeriche per la normalizzazione. Ora procederemo con la pulizia dei dati mancanti, 
# il preprocessing e l'unione del dataset delle recensioni con il dataset dei film.


# Controllo dei valori mancanti nel dataset dei film
print(movies_df.isnull().sum())

# Controllo dei valori mancanti nel dataset delle recensioni
print(ratings_df.isnull().sum())

# Rimozione delle colonne con più del 50% di valori mancanti (se presenti)
movies_df = movies_df.dropna(thresh=len(movies_df) * 0.5, axis=1)

# Riempimento dei valori mancanti in alcune colonne (se necessario)
movies_df['budget'] = movies_df['budget'].fillna(movies_df['budget'].median())
movies_df['revenue'] = movies_df['revenue'].fillna(movies_df['revenue'].median())

# Unione dei due dataset sulla colonna 'movieId'
df_merged = pd.merge(movies_df, ratings_df, on='movieId', how='inner')

# Controllo delle dimensioni del dataset unito
print(f"Dimensioni del dataset unito: {df_merged.shape}")

from sklearn.preprocessing import MinMaxScaler

# Normalizzazione delle recensioni (valori da 1 a 5)
scaler = MinMaxScaler()
df_merged['rating'] = scaler.fit_transform(df_merged[['rating']])

#Feature Engineering (se necessario):
#Se desideri, puoi creare nuove feature utili per il modello di machine learning. Ad esempio, puoi trasformare
#alcune colonne in categorie (come il genere del film) e applicare tecniche di one-hot encoding o label encoding.

# One-hot encoding delle colonne categoriche (ad esempio 'genres')
df_merged = pd.get_dummies(df_merged, columns=['genres_list'], drop_first=True)

#Divisione dei dati in set di addestramento e test
from sklearn.model_selection import train_test_split

# Divisione dei dati in input (X) e target (y)
X = df_merged.drop(columns=['rating', 'userId', 'movieId'])  # Rimuovi le colonne non necessarie
y = df_merged['rating']

# Divisione in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df_merged.head())

