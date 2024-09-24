import pandas as pd

# Caricamento dei file CSV
df_rat = pd.read_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/ratings_small.csv", low_memory=False)
df_movie = pd.read_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/movies_metadata_50_perc.csv", low_memory=False)

# Trova gli ID comuni tra i due dataset
unici_df_rat = pd.Series(df_rat['movieId'].unique())
comuni = df_movie['movieId'].isin(unici_df_rat)  # Film presenti in entrambi i dataset

# Filtra i film comuni in df_movie
df_movie_comuni = df_movie[comuni]

# Filtra le recensioni comuni in df_rat
presenti_rec = df_rat['movieId'].isin(df_movie_comuni['movieId'])
df_rat_comuni = df_rat[presenti_rec]

# Salva i nuovi dataset filtrati
df_movie_comuni.to_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/movies_metadata_comuni.csv", index=False)
df_rat_comuni.to_csv("/Users/giacomovecchio/Desktop/corso crif/progetto/ratings_comuni.csv", index=False)

# Conteggi per verificare
print(f"Numero di film comuni nel dataset movies: {len(df_movie_comuni)}")
print(f"Numero di recensioni comuni nel dataset ratings: {len(df_rat_comuni)}")
