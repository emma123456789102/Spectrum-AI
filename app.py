import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
from flask import Flask
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np

app = Flask(__name__)

cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autismdiagnosistool-default-rtdb.europe-west1.firebasedatabase.app'
})

model = joblib.load('autism_best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict')
def predict():
    ref = db.reference('autism_responses')
    entries = ref.get()

    results = {}
    for key, entry in entries.items():
        features = np.array([entry['age'], entry['income']]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        results[key] = int(prediction)

        result_ref = db.reference(f"autism_responses/{key}/result")
        result_ref.set(int(prediction))

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data submitted by the user
    communication = request.form['communication']
    developmental = request.form['developmental']
    eye_contact = request.form['eye_contact']
    interest_in_children = request.form['interest_in_children']
    pretend_play = request.form['pretend_play']
    repetitive_movements = request.form['repetitive_movements']
    response_to_name = request.form['response_to_name']
    routine_changes = request.form['routine_changes']
    sensory_sensitivities = request.form['sensory_sensitivities']
    social_cues = request.form['social_cues']
    specific_interests = request.form['specific_interests']
    
    # Create a DataFrame from the form data
    data = {
        'communication': [communication],
        'developmental': [developmental],
        'eye_contact': [eye_contact],
        'interest_in_children': [interest_in_children],
        'pretend_play': [pretend_play],
        'repetitive_movements': [repetitive_movements],
        'response_to_name': [response_to_name],
        'routine_changes': [routine_changes],
        'sensory_sensitivities': [sensory_sensitivities],
        'social_cues': [social_cues],
        'specific_interests': [specific_interests],
    }
    
    df = pd.DataFrame(data)

    # Encode categorical features using LabelEncoder
    for column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Make the prediction using the trained model
    prediction = model.predict(df)

    # Map the result to a human-readable format
    result = 'Autism Diagnosed' if prediction[0] == 1 else 'No Autism'

    return render_template('result.html', result=result)

# For getting data from Firebase (example)
@app.route('/fetch_data')
def fetch_data():
    # Example: Fetching the data from Firestore collection
    collection_ref = db.collection('autism_data')  # Change to your collection name
    docs = collection_ref.stream()

    # Store the fetched data in a list for rendering or processing
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    
    return render_template('data_display.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
