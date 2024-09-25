import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Caricamento dei dataset
movies_df = pd.read_csv('ML Project\\Movies Dataset\\csv giusti\\movies_metadata_finale_small.csv')
ratings_df = pd.read_csv('ML Project\\Movies Dataset\\csv giusti\\ratings_small_finale.csv')
# ratings_df['ratings']=ratings_df['ratings']*2
# ratings_df['ratings']=ratings_df['ratings'].astype(int)
# Preprocessing dei dati
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

# One-Hot Encoding per le colonne categoriali
encoder = OneHotEncoder(sparse_output=False)
encoded_genres = encoder.fit_transform(merged_df['genres_list'].values.reshape(-1, 1))# qui è stato usato solo il genere, ignoranti lingua,paese di produzione, case ecc

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
model.add(Dense(10, activation='softmax'))# Output per la valutazione

# Compilazione del modello
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')#learning rate?

# Addestramento del modello
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

#A questo punto c'è una parte di prediction per i film non visti. Prima si potrebbe fare una parte di prediction solo per i film visti da ogni singolo utente dati dal training per valutare l'accuratezza

def round_to_nearest_int(x):
    return np.round(x)  # Arrotonda al più vicino 0.5

# Funzione di raccomandazione
def recommend_movies(user_id, ratings_df, model, movies_df, num_recommendations=20):
    # Ottieni i film valutati dall'utente
    user_movies = ratings_df[ratings_df['userId'] == user_id]
    seen_movie_ids = user_movies['movieId'].unique()
    
    # Stampa i film visti dall'utente, il titolo e il voto
    #si potrebbero stampare solo i 10 film con valutazioni più alte, descending
    if not user_movies.empty:
        print(f"Film visti dall'utente {user_id}:")
        for index, row in user_movies.iterrows():
            title = movies_df[movies_df['movieId'] == row['movieId']]['title'].values[0]
            print(f" - {title} (Rating: {row['rating']})")
            #print(f" - {title} (Generi: {row['genres_list']})")#aggiunto stamattina per il controllo dei generi stampati, controllare
    else:
        print(f"L'utente {user_id} non ha visto alcun film.")

    # Prendi i film che l'utente non ha visto
    unseen_movies = movies_df[~movies_df['movieId'].isin(seen_movie_ids)]
    
    # Prepara i dati per le previsioni
    unseen_genres_encoded = encoder.transform(unseen_movies['genres_list'].values.reshape(-1, 1))
    
    # Crea un dataframe per unire le caratteristiche, escludendo userId
    unseen_X = pd.DataFrame(unseen_genres_encoded, columns=encoder.get_feature_names_out())
    
    # Aggiungi solo le caratteristiche necessarie (senza userId)
    unseen_X = pd.concat([
        unseen_X,
        unseen_movies[['budget', 'runtime', 'popularity', 'vote_average']].reset_index(drop=True)
    ], axis=1)

    # Assicurati di avere solo le colonne utilizzate in X_train
    unseen_X = unseen_X.reindex(columns=X_train.columns, fill_value=0)  # Riordina e riempi eventuali colonne mancanti
    # Normalizza i dati per le previsioni
    unseen_X_scaled = scaler.transform(unseen_X)

    # Previsione delle valutazioni
    predicted_ratings = model.predict(unseen_X_scaled)
    
    # Limita le valutazioni previste tra 0 e 5
    #predicted_ratings = np.clip(predicted_ratings, 0, 5)
    
    # Arrotonda le previsioni al più vicino multiplo di 0.5
    #predicted_ratings = round_to_nearest_half(predicted_ratings)

    # Aggiungi le previsioni al dataframe dei film
    unseen_movies['predicted_rating'] = predicted_ratings.flatten()
    
    # Raccomanda i film con le valutazioni più alte
    recommended_movies = unseen_movies.sort_values(by='predicted_rating', ascending=False).head(num_recommendations)
    return recommended_movies[['title', 'predicted_rating']]

def previsioni_esistenti(test, etichetta, modello):

    # Previsione delle valutazioni
    predicted_ratings = modello.predict(test)
    predicted_ratings=round_to_nearest_int(predicted_ratings)
    predicted_ratings = np.clip(predicted_ratings, 0, 10)#serve solo per un numero basso di epoche
    cm = confusion_matrix(etichetta, predicted_ratings)
    
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Aggiungere etichette
    plt.xlabel('Voti previsti')
    plt.ylabel('Voti reali')
    plt.title('Matrice di Confusione')
    plt.show()
    

previsioni_esistenti(X_test_scaled, y_test, model)
# Raccomandazione per un utente
recommended = recommend_movies(user_id=1, ratings_df=ratings_df, model=model, movies_df=movies_df)
print(recommended)
