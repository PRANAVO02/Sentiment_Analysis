import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense,Dropout

# Load dataset
name_column = ['id', 'entity', 'target', 'Tweet content']
try:
    df = pd.read_csv('twitter_training.csv', names=name_column)
except FileNotFoundError:
    print("Error: File 'twitter_training.csv' not found.")
    exit()

# Data preprocessing
df = df.drop(columns=['id', 'entity'], axis=1)
df.dropna(inplace=True)

# Text preprocessing
nltk.download('stopwords')
ps = PorterStemmer()
stops = set(stopwords.words('english'))


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    token = text.split()
    token = [ps.stem(word) for word in token if word not in stops]
    return ' '.join(token)


df['Tweet content'] = df['Tweet content'].apply(preprocess_text)

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['Tweet content'])
X = tokenizer.texts_to_sequences(df['Tweet content'])
X = pad_sequences(X, maxlen=100)

# Convert target to numeric
y = pd.get_dummies(df['target']).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Function to build and compile models
def build_and_compile_model(model_type):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))

    if model_type == 'LSTM':
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == 'GRU':
        model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
    elif model_type == 'RNN':
        model.add(SimpleRNN(128, dropout=0.2))
    elif model_type == 'CNN':
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(GlobalMaxPooling1D())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and evaluate each model
models = ['LSTM', 'GRU', 'RNN', 'CNN']
for model_type in models:
    print(f"\nTraining {model_type} model...")
    model = build_and_compile_model(model_type)
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test_classes = y_test.argmax(axis=1)

    print(f"\n--- {model_type} Model Results ---")
    print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
    print("\nClassification Report:\n", classification_report(y_test_classes, y_pred_classes))