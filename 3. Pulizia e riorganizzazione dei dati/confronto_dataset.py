import pandas as pd

df_rat=pd.read_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\ratings_small_CLEAN.csv",low_memory=False)
df_movie=pd.read_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN_ID.csv",low_memory=False)
unici_df_rat = pd.Series(df_rat['movieId'].unique())
presenti = df_movie['movieId'].isin(unici_df_rat)
# Valori presenti
valori_presenti = df_movie[presenti]

# Valori non presenti
valori_non_presenti = df_movie[~presenti]

num_presenti = presenti.sum()  # Conta i True
num_non_presenti = (~presenti).sum()  # Conta i False
perc_pres= num_presenti/(num_presenti+num_non_presenti)
print(f"Film presenti in entrambi {num_presenti}")
print(f"Film non recensiti {num_non_presenti}")
print(f"La percentuale di film presenti è {perc_pres}")



presenti_rec = unici_df_rat.isin(df_movie['movieId'])
# Valori presenti
valori_presenti_rec = unici_df_rat[presenti_rec]

# Valori non presenti
valori_non_presenti_rec = unici_df_rat[~presenti_rec]

num_presenti_rec = presenti_rec.sum()  # Conta i True
num_non_presenti_rec = (~presenti_rec).sum()  # Conta i False
perc_pres_rec= num_presenti_rec/(num_presenti_rec+num_non_presenti_rec)
print(f"Film presenti in entrambi {num_presenti_rec}")
print(f"Film recesniti non presenti in metadata{num_non_presenti_rec}")
print(f"La percentuale di film presenti in recensione è {perc_pres_rec}")