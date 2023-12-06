import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Load your CSV data
data = pd.read_csv(r'D:\urldata.csv')

# Convert 'result' column to 'label' with 0 and 1
data['label'] = data['label'].map({'benign': 0, 'malicious': 1})

# Split the data into features and labels
X = data['url']
y = data['label']

# Initialize a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Reduce dimensionality using Truncated SVD
n_components = 100
svd = TruncatedSVD(n_components=n_components)
X_tfidf = svd.fit_transform(X_tfidf)

# Standardize the data
scaler = StandardScaler()
X_tfidf = scaler.fit_transform(X_tfidf)

# Build a simple ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_components,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_tfidf, y, epochs=10, batch_size=32)


# Function to predict URL
def predict_url(url):
    # Transform the input URL using the same vectorizers and preprocessing steps
    url_tfidf = tfidf_vectorizer.transform([url])
    url_tfidf = svd.transform(url_tfidf)
    url_tfidf = scaler.transform(url_tfidf)

    # Make the prediction
    prediction = model.predict(url_tfidf)

    # Output the result
    if prediction[0][0] >= 0.5:
        return f"The URL '{url}' is predicted to be malicious."
    else:
        return f"The URL '{url}' is predicted to be benign."


# Allow the user to enter URLs multiple times
while True:
    user_url = input("Enter the URL to predict (type 'exit' to stop): ")

    if user_url.lower() == 'exit':
        break

    result = predict_url(user_url)
    print(result)
