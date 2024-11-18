import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dense
from sklearn.model_selection import train_test_split
import re

# Step 1: Load Data from CSV
df = pd.read_csv('twitter_dataset1.csv')

# Step 2: Function to Replace Emojis with Text
def replace_emojis(text):
    text = re.sub(r":\)", " happy ", text)  # Replace ":)" with "happy"
    text = re.sub(r":\(", " sad ", text)    # Replace ":(" with "sad"
    return text

# Apply the emoji replacement function to the dataset
df['text'] = df['text'].apply(replace_emojis)

# Step 3: Encode Labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # Encoding 'threatening' -> 0, 'playful' -> 1, 'neutral' -> 2

# Step 4: Text Preprocessing (Tokenization and Padding)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])

# Pad the sequences to the same length
X = pad_sequences(X, padding='post', maxlen=50)

# Labels
y = df['label'].values

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the CNN Model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=50))  # Embedding layer to convert words to vectors
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))  # Convolutional layer to extract features
model.add(MaxPooling1D(pool_size=4))  # Max pooling to downsample
model.add(Flatten())  # Flatten the output to feed into fully connected layers
model.add(Dense(3, activation='softmax'))  # Output layer with 3 classes: threatening, playful, neutral

# Step 7: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Step 9: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Step 10: Loop to allow User Input and Predict the Label
while True:
    user_input = input("Enter a text to classify (or type 'exit' to stop): ")

    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    # Preprocess the input text (replace emojis, tokenization, and padding)
    user_input = replace_emojis(user_input)  # Replace emojis with text equivalents
    new_sequences = tokenizer.texts_to_sequences([user_input])  # Convert the text to sequences
    new_sequences = pad_sequences(new_sequences, padding='post', maxlen=50)  # Pad the sequence

    # Make a prediction
    predictions = model.predict(new_sequences)

    # Decode predictions
    predicted_label = label_encoder.inverse_transform(
        predictions.argmax(axis=1))  # Get the label with the highest probability

    # Output the prediction
    print(f"Text: {user_input}")
    print(f"Predicted Label: {predicted_label[0]}")
    print("-" * 50)
