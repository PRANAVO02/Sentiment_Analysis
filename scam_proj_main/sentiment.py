import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

print("Modules loaded")

# Load dataset
name_column = ['id', 'entity', 'target', 'Tweet content']
try:
    df = pd.read_csv('twitter_training.csv', names=name_column)
except FileNotFoundError:
    print("Error: File 'twitter_training.csv' not found.")
    exit()

# Data inspection
df.info()

# Drop unnecessary columns and missing values
df = df.drop(columns=['id', 'entity'], axis=1)
df.dropna(inplace=True)

# Plot target distribution
count = df['target'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=count.index, y=count.values, hue=count.index, palette='viridis', dodge=False)
plt.title('Target Counts')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

# Text preprocessing
ps = PorterStemmer()
stops = set(stopwords.words('english'))

def preprocessing_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    token = text.split()
    token = [ps.stem(word) for word in token if word not in stops]
    return ' '.join(token)

df['Tweet content'] = df['Tweet content'].apply(preprocessing_text)

# TF-IDF Vectorization
tf = TfidfVectorizer(max_features=5000)
x = tf.fit_transform(df['Tweet content'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

# Model training and evaluation
models = {
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=model.classes_)
    plt.title(f'{name} Confusion Matrix')
    plt.show()