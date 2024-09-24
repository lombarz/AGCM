import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Carica i due dataset
movies = pd.read_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\csv giusti\\movies_metadata_comuni_small.csv')
ratings = pd.read_csv('C:\\Users\\Utilizzatore\\Desktop\\Academy\\ML Project\\Movies Dataset\\csv giusti\\ratings_comuni_small.csv')

# Unisci i dataset tramite 'movieId'
merge_df = pd.merge(movies, ratings, on='movieId')

# Preprocessamento
# Rimuoviamo eventuali valori mancanti
merge_df = merge_df.dropna()

# Conversione delle colonne categoriche
categorical_columns = ['original_language', 'genres_list', 'production_companies_list', 
                       'production_countries_list', 'spoken_languages_list']
label_encoder = LabelEncoder()
for col in categorical_columns:
    merge_df[col] = label_encoder.fit_transform(merge_df[col])

# Definisci le caratteristiche (X) e il target (y)
X = merge_df[[ 'rating','budget', 'popularity', 'revenue', 'runtime', 
              'vote_average', 'vote_count', 'release_year'] + categorical_columns]
y = merge_df['userId']

# Converti 'userId' in formato numerico e one-hot encoding
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
y = to_categorical(y)

# Standardizza le caratteristiche numeriche
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividi il dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruisci il modello di rete neurale
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))  # Dropout del 50% per prevenire overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Output per multi-class classification,possibile sostiuire con 10

# Compila il modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestra il modello
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Visualizza l'andamento della funzione di perdita
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Perdita di training')
plt.plot(history.history['val_loss'], label='Perdita di validazione')
plt.title('Funzione di perdita con Dropout')
plt.xlabel('Epoca')
plt.ylabel('Perdita')
plt.legend()
plt.show()
