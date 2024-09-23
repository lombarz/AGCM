import pandas as pd

df1=pd.read_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN.csv",low_memory=False)
df2=pd.read_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\links.csv",low_memory=False)
df1 = df1.drop_duplicates(subset='id')
df2 = df2.drop_duplicates(subset='movieId')
df2 = df2.drop_duplicates(subset='tmdbId')
mappa_id = df2.set_index('tmdbId')['movieId']

# Creiamo una nuova colonna con il nuovo ID usando .map()
df1['movieId'] = df1['id'].map(mappa_id)

# Eliminiamo la colonna id_originale
df1 = df1.drop(columns=['id'])

df1.to_csv("C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN_ID.csv",index=False)