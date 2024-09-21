import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\movies_metadata_CLEAN.csv')
print(df.head())
print(df.info())  # Vedi il tipo di dati
print(df.describe())  # Statistiche di base

le = LabelEncoder()#converte stringfhe e oggetti in numeri, utile per la comprensione della macchina
for col in df.columns:
    if df[col].dtype == 'object':  # Controlla se il tipo di dato è 'object' (tipicamente stringa)
        df[col] = le.fit_transform(df[col])
    
scaler = StandardScaler()
df_clean= df.dropna()#elimino le righe con dati Nan
df_scaled = scaler.fit_transform(df_clean)

# Definisci il numero di cluster
kmeans = KMeans(n_clusters=4, random_state=42)

# Applica il modello ai dati normalizzati
kmeans.fit(df_scaled)

# Ottieni le etichette dei cluster
labels = kmeans.labels_

# Aggiungi le etichette al DataFrame pulito
df_clean['cluster'] = labels
print(df_clean.head())
print(df_clean.tail())

#rendo i dati 2d per uno scatter plot
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Crea lo scatter plot delle prime 2 componenti
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_clean['cluster'], cmap='viridis')
plt.title('Movies- Cluster Plot')
plt.xlabel('Prima componente principale')
plt.ylabel('Seconda componente principale')
plt.show()