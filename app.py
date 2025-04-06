from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Autism Questionnaire App!"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    # Perform the analysis here (mock logic as an example)
    result = {"diagnosis": "moderate likelihood of autism traits"}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

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
