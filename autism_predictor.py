# autism_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Load dataset
def load_data():
    url = "https://raw.githubusercontent.com/Chandana2397/Autism-Spectrum-Prediction/main/autism_screening.csv"
    df = pd.read_csv(url)
    return df

# Preprocess data
def preprocess_data(df):
    df = df.drop(['jundice', 'austim', 'contry_of_res', 'ethnicity', 'used_app_before', 'age_desc', 'relation', 'Class/ASD'], axis=1)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical columns
    label_cols = ['gender', 'screening_score']
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop('result', axis=1)
    y = df['result']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Train and save models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, 'models/logistic_regression.pkl')

    # SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'models/svm.pkl')

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/random_forest.pkl')

    # Neural Network
    nn = Sequential()
    nn.add(Dense(32, input_dim=X.shape[1], activation='relu'))
    nn.add(Dense(16, activation='relu'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)
    nn.save('models/neural_network.h5')

    print("Models trained and saved in 'models/' directory.")

if __name__ == '__main__':
    df = load_data()
    X, y, scaler = preprocess_data(df)
    train_models(X, y)
    joblib.dump(scaler, 'models/scaler.pkl')
