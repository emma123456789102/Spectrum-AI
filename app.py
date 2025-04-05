import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load your trained machine learning model
model = joblib.load('path_to_your_model.pkl')

# Initialize encoders and scalers
label_encoder = LabelEncoder()

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
