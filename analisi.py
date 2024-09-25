import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Caricamento dei dataset
movies_df = pd.read_csv('/Users/giacomovecchio/Desktop/corso crif/progetto2/db/movies_metadata_small_finale.csv')
ratings_df = pd.read_csv('/Users/giacomovecchio/Desktop/corso crif/progetto2/db/ratings_small_finale.csv')

# Preprocessing dei dati
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

# One-Hot Encoding per le colonne categoriali
encoder = OneHotEncoder(sparse_output=False)
encoded_genres = encoder.fit_transform(merged_df['genres_list'].values.reshape(-1, 1))

# Creazione di un dataframe di input per il modello
X = pd.concat([
    pd.DataFrame(encoded_genres, columns=encoder.get_feature_names_out()),
    merged_df[['budget', 'runtime', 'popularity', 'vote_average', 'userId']]
], axis=1)

# Target: valutazione del film
y = merged_df['rating'].values

# Suddivisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalizzazione delle caratteristiche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Costruzione del modello
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  

# Compilazione del modello
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Addestramento del modello
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# Funzione di raccomandazione
def recommend_movies(user_id, ratings_df, model, movies_df, num_recommendations=20):
    user_movies = ratings_df[ratings_df['userId'] == user_id]
    seen_movie_ids = user_movies['movieId'].unique()
    
    if not user_movies.empty:
        print(f"Film visti dall'utente {user_id}:")
        for index, row in user_movies.iterrows():
            title = movies_df[movies_df['movieId'] == row['movieId']]['title'].values[0]
            print(f" - {title} (Rating: {row['rating']})")
    else:
        print(f"L'utente {user_id} non ha visto alcun film.")

    unseen_movies = movies_df[~movies_df['movieId'].isin(seen_movie_ids)]
    unseen_genres_encoded = encoder.transform(unseen_movies['genres_list'].values.reshape(-1, 1))
    
    unseen_X = pd.DataFrame(unseen_genres_encoded, columns=encoder.get_feature_names_out())
    unseen_X = pd.concat([
        unseen_X,
        unseen_movies[['budget', 'runtime', 'popularity', 'vote_average']].reset_index(drop=True)
    ], axis=1)

    unseen_X = unseen_X.reindex(columns=X_train.columns, fill_value=0)
    unseen_X_scaled = scaler.transform(unseen_X)

    predicted_ratings = model.predict(unseen_X_scaled)
    unseen_movies['predicted_rating'] = predicted_ratings.flatten()
    
    recommended_movies = unseen_movies.sort_values(by='predicted_rating', ascending=False).head(num_recommendations)
    return recommended_movies[['title', 'predicted_rating']]

# Funzione per arrotondare le predizioni al numero intero più vicino
def round_to_nearest_int(predictions):
    return np.rint(predictions).astype(int)

# Funzione per le previsioni esistenti con matrice di confusione
# Funzione per le previsioni esistenti con matrice di confusione evidenziando la diagonale
def previsioni_esistenti(test, etichetta, modello):
    # Previsione delle valutazioni
    predicted_ratings = modello.predict(test)
    
    # Arrotonda al numero intero più vicino
    predicted_ratings = round_to_nearest_int(predicted_ratings)
    
    # Assicura che le previsioni siano tra 0 e 10
    predicted_ratings = np.clip(predicted_ratings, 0, 10)
    
    # Calcolo della matrice di confusione
    cm = confusion_matrix(etichetta, predicted_ratings)
    
    # Calcolo dell'accuratezza
    accuracy = accuracy_score(etichetta, predicted_ratings)
    print(f'Accuratezza: {accuracy * 100:.2f}%')

    # Creazione di una maschera per evidenziare la diagonale
    mask = np.eye(cm.shape[0], dtype=bool)
    
    # Visualizzazione della matrice di confusione con evidenziazione della diagonale
    plt.figure(figsize=(10, 7))
    
    # Heatmap con colori diversi per la diagonale
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor='black', square=True)
    
    # Applicazione di un colore diverso per la diagonale
    sns.heatmap(cm, mask=~mask, annot=False, cmap="Reds", cbar=False, linewidths=0.5, linecolor='black', square=True)

    # Aggiungere etichette
    plt.xlabel('Voti previsti')
    plt.ylabel('Voti reali')
    plt.title(f'Matrice di Confusione\nAccuratezza: {accuracy * 100:.2f}%')
    
    plt.show()

previsioni_esistenti(X_test_scaled, y_test, model)

recommended = recommend_movies(user_id=1, ratings_df=ratings_df, model=model, movies_df=movies_df)
print(recommended)
