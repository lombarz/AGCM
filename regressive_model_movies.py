import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# Caricamento dei dataset
movies_df = pd.read_csv('ML Project\\Movies Dataset\\csv giusti\\movies_metadata_finale_small.csv')
ratings_df = pd.read_csv('ML Project\\Movies Dataset\\csv giusti\\ratings_small_finale.csv')

while True:
    choiche=input("Benvenuto, scegli un'opzione:\n 1= Consigli film per un nuovo utente \n 2= Consigli film per un utente esistente\n 3=esci")
    if choiche=='1':
        max=ratings_df['userId'].max()
        id_request=max+1
        sorted_movies = movies_df.sort_values(by='vote_count', ascending=False)
        count=0
        while True:
          for index, row in sorted_movies.iterrows():
            title = row['title']
            vote_count = row['vote_count']
            # Chiedi all'utente
            response = input(f"Hai visto '{title}'? (s/n): ").strip().lower()

            # Puoi registrare la risposta in un elenco o elaborare ulteriormente
            if response == 's':
               vote=int(input('Dagli un punteggio da 0 a 10(intero):'))#continuiamo da qui domani
               if vote in range (0,11):
                 count+=1
                 if count==10:
                   break
                 else:
                   continue
               else:
                 print("il voto non è valido")
            elif response == 'n':
               continue
            else:
               print("Risposta non valida. Si prega di rispondere con 's' o 'n'.")
    elif choiche=='2':
        id=int(input("Inserisci l'id utente:"))
        if id in merged_df['userId']:
            id_request=id
        else:
            print("Utente non presente")
    elif choiche == '3':
        print('Arrivederci!')
        break
    else:
        print('Scelta non valida')
# Preprocessing dei dati
merged_df = pd.merge(ratings_df, movies_df, on='movieId')
# One-Hot Encoding per le colonne categoriali
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
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
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Addestramento del modello
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

def round_to_nearest_int(x):
    return np.round(x)

# Funzione per proporre i film
def recommend_movies(user_id, ratings_df, model, movies_df, encoder, scaler, num_recommendations=20):
    # Assicurati che la colonna 'genres_list' sia una lista di stringhe
    if movies_df['genres_list'].dtype == 'object':
        movies_df['genres_list'] = movies_df['genres_list'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Ottieni i film valutati dall'utente
    user_movies = ratings_df[ratings_df['userId'] == user_id]
    seen_movie_ids = user_movies['movieId'].unique()
    
    # Stampa i film visti dall'utente e i generi
    if not user_movies.empty:
        print(f"Film visti dall'utente {user_id}:")
        top_rated_movies = user_movies.sort_values(by='rating', ascending=False).head(10)
        
        # Colleziona i generi visti
        seen_genres = []
        
        for index, row in top_rated_movies.iterrows():
            title = movies_df[movies_df['movieId'] == row['movieId']]['title'].values[0]
            genres = movies_df[movies_df['movieId'] == row['movieId']]['genres_list'].values[0]
            seen_genres.extend(genres)
            print(f" - {title} (Rating: {row['rating']})")
            print(f"   Generi: {genres}")
        
        # Contare i generi più comuni tra i film visti
        most_common_seen_genres = Counter(seen_genres).most_common(3)
        print("\nTre generi più presenti nei film visti:")
        for genre, count in most_common_seen_genres:
            print(f" - {genre}: {count} volte")
    else:
        print(f"L'utente {user_id} non ha visto alcun film.")

    # Prendi i film che l'utente non ha visto
    unseen_movies = movies_df[~movies_df['movieId'].isin(seen_movie_ids)]

    # Appiattire i generi in una stringa per l'encoder
    unseen_movies.loc[:, 'genres_str'] = unseen_movies['genres_list'].apply(lambda x: ', '.join(x))

    # Prepara i dati per le previsioni
    unseen_genres_encoded = encoder.transform(unseen_movies['genres_str'].values.reshape(-1, 1))
    unseen_X = pd.DataFrame(unseen_genres_encoded, columns=encoder.get_feature_names_out())
    
    unseen_X = pd.concat([unseen_X, unseen_movies[['budget', 'runtime', 'popularity', 'vote_average']].reset_index(drop=True)], axis=1)
    unseen_X = unseen_X.reindex(columns=X_train.columns, fill_value=0)
    
    unseen_X_scaled = scaler.transform(unseen_X)

    # Previsione delle valutazioni
    predicted_ratings = model.predict(unseen_X_scaled)
    predicted_ratings = np.clip(predicted_ratings, 0, 10)
    
    unseen_movies['predicted_rating'] = predicted_ratings.flatten()
    
    # Raccomanda i film con le valutazioni più alte
    recommended_movies = unseen_movies.sort_values(by='predicted_rating', ascending=False).head(num_recommendations)
    
    # Colleziona i generi raccomandati
    recommended_genres = []
    
    for index, row in recommended_movies.iterrows():
        genres = row['genres_list']
        recommended_genres.extend(genres)

    # Contare i generi più comuni tra i film raccomandati
    most_common_recommended_genres = Counter(recommended_genres).most_common(3)

    # Stampa i risultati
    print("\nTre generi più presenti nei film raccomandati:")
    for genre, count in most_common_recommended_genres:
        print(f" - {genre}: {count} volte")

    return recommended_movies[['title', 'predicted_rating', 'genres_list']]

def previsioni_esistenti(test, etichetta, modello):
    # Previsione delle valutazioni
    predicted_ratings = modello.predict(test)
    predicted_ratings = round_to_nearest_int(predicted_ratings)
    predicted_ratings = np.clip(predicted_ratings, 0, 10)
    cm = confusion_matrix(etichetta, predicted_ratings)
    accuracy = accuracy_score(etichetta, predicted_ratings)
    print(f"Accuracy: {accuracy:.2f}")
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Aggiungere etichette
    plt.ylabel('Voti previsti')
    plt.xlabel('Voti reali')
    plt.title('Matrice di Confusione')
    plt.show()

# Valutazione della performance
previsioni_esistenti(X_test_scaled, y_test, model)

# Raccomandazione per un utente
recommended = recommend_movies(user_id=id_request, ratings_df=ratings_df, model=model, movies_df=movies_df, encoder=encoder, scaler=scaler)
print(recommended)
