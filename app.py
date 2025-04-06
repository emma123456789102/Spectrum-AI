from flask import Flask, jsonify, request
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase
cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autismdiagnosistool-default-rtdb.europe-west1.firebasedatabase.app'
})

# Load the model and scaler
model = joblib.load('autism_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route
@app.route('/')
def home():
    return "Welcome to the Autism Questionnaire App!"

# Analyze route (mock logic for now)
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    result = {"diagnosis": "moderate likelihood of autism traits"}
    return jsonify(result)

# Prediction route
@app.route('/predict', methods=['GET'])
def predict():
    ref = db.reference('autism_responses')
    entries = ref.get()

    results = {}
    if entries:
        for key, entry in entries.items():
            features = np.array([entry.get('age', 0), entry.get('income', 0)]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            results[key] = int(prediction)

            # Save prediction result back to Firebase
            result_ref = db.reference(f"autism_responses/{key}/result")
            result_ref.set(int(prediction))

    return jsonify(results)

# Start the app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
