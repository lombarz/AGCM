import pandas as pd

df=pd.read_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\ratings.csv",low_memory=False)
df_ratings_clean=df
#togliamo eventuali duplicati e copie inutili
df_ratings_clean.drop_duplicates(inplace=True)
df_ratings_clean.drop(columns=['timestamp'],inplace=True)
print(df_ratings_clean.head())
print(df_ratings_clean.info())  # Vedi il tipo di dati
print(df_ratings_clean.describe())  # Statistiche di base

df_ratings_clean.to_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\ratings_CLEAN.csv",index=False)