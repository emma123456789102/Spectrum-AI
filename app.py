
# --- Deployment Instructions ---
print("\nModel saved! To deploy, use Flask or FastAPI.")
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle cross-origin requests
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('autism_prediction_model_advanced.pkl')
scaler = joblib.load('scaler.pkl')  # Save the scaler during training and load it here

# Define the feature columns (Replace with actual feature names from your dataset)
feature_columns = ['A1_score', 'A2_score', 'age', 'gender']  # Update accordingly

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

@app.route('/')
def home():
    return "Autism Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Your code logic here
        pass  # Placeholder for the actual code
    except Exception as e:
        return jsonify({'error': str(e)})


# Define a simple scoring function
def calculate_autism_score(answers):
    # Assuming answers is a list of 10 responses (1 for Yes, 0 for No)
    score = sum(answers)
    
    # Define risk categories
    if score >= 6:
        result = "High likelihood of autism, consider consulting a specialist."
    elif score >= 3:
        result = "Moderate likelihood, monitoring recommended."
    else:
        result = "Low likelihood of autism."
    
    return score, result

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Receive data from frontend
    answers = data.get("answers", [])  # Expecting a list of numbers
    
    if len(answers) != 10:  # Ensure correct number of answers
        return jsonify({"error": "Invalid number of answers, expected 10."}), 400

    score, message = calculate_autism_score(answers)
    
    return jsonify({"score": score, "message": message})

if __name__ == "__main__":
    app.run(debug=True)
@app.route("/predict_model", methods=["POST"])
def predict_model():
    try:
        # Receive JSON data from frontend
        data = request.get_json()
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data], columns=feature_columns)
        
        # Standardize the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        result = "Autistic" if prediction[0] == 1 else "Not Autistic"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

