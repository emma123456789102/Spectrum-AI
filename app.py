from flask import Flask, jsonify
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autismdiagnosistool-default-rtdb.europe-west1.firebasedatabase.app'
})

# Load the model and scaler
model = joblib.load('autism_best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    ref = db.reference('autism_responses')
    entries = ref.get()

    results = {}
    if entries:
        for key, entry in entries.items():
            # Assuming 'age' and 'income' exist
            features = np.array([entry.get('age', 0), entry.get('income', 0)]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            results[key] = int(prediction)

            # Save prediction result back to Firebase
            result_ref = db.reference(f"autism_responses/{key}/result")
            result_ref.set(int(prediction))

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
