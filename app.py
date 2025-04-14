# streamlit_app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
logreg = joblib.load('models/logistic_regression.pkl')
svm = joblib.load('models/svm.pkl')
rf = joblib.load('models/random_forest.pkl')
nn = load_model('models/neural_network.h5')
scaler = joblib.load('models/scaler.pkl')

# 30-question autism questionnaire
questions = [
    "Did the individual have a delay in speaking their first words?",
    "Does the individual avoid group activities or social gatherings?",
    "Does the individual dislike certain textures of food or clothing?",
    "Does the individual engage in repetitive movements (e.g., hand flapping, rocking, spinning)?",
    "Does the individual engage in self-soothing behaviors (e.g., tapping, humming, hand-waving)?",
    "Does the individual enjoy sensory-seeking behaviors (e.g., spinning, jumping, touching different textures)?",
    "Does the individual experience frequent emotional outbursts or meltdowns?",
    "Does the individual get extremely bothered by certain noises?",
    "Does the individual have an intense focus on specific topics or objects?",
    "Does the individual have difficulty transitioning from one activity to another?",
    "Does the individual have difficulty understanding other people’s feelings?",
    "Does the individual have trouble adapting to changes in social rules?",
    "Does the individual insist on following the same routine every day and get upset by changes?",
    "Does the individual line up toys or objects in a specific order instead of playing with them?",
    "Does the individual make appropriate eye contact during conversations?",
    "Does the individual overreact or underreact to loud sounds, bright lights, or strong smells?",
    "Does the individual prefer communicating through writing, pictures, or gestures rather than speech?",
    "Does the individual prefer to play alone rather than with other children?",
    "Does the individual repeat certain questions or phrases even when the conversation has moved on?",
    "Does the individual repeat words or phrases (echolalia) without understanding their meaning?",
    "Does the individual respond when their name is called?",
    "Does the individual show an unusual attachment to specific objects?",
    "Does the individual show extreme distress when plans change unexpectedly?",
    "Does the individual speak in a monotone or robotic voice?",
    "Does the individual struggle to hold two-way conversations?",
    "Does the individual struggle to make or maintain friendships?",
    "Does the individual struggle to understand or follow multi-step instructions?",
    "Does the individual struggle to understand sarcasm or jokes?",
    "Does the individual struggle with managing frustration or disappointment?",
    "Does the individual take things very literally?"
]

answer_options = ["Never", "Rarely", "Sometimes", "Often"]
answer_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}

def main():
    st.title("🧠 Spectrum-AI - Autism Questionnaire Predictor")
    st.write("Answer the following questions to see predictions from 4 different models.")

    user_input = []
    for q in questions:
        answer = st.radio(q, answer_options, horizontal=True)
        user_input.append(answer_map[answer])

    user_input = np.array([user_input])
    user_input_scaled = scaler.transform(user_input)

    if st.button("Predict"):
        st.subheader("🧪 Model Predictions")

        pred_log = logreg.predict(user_input_scaled)[0]
        pred_svm = svm.predict(user_input_scaled)[0]
        pred_rf = rf.predict(user_input_scaled)[0]
        pred_nn = (nn.predict(user_input_scaled)[0][0] > 0.5).astype(int)

        def label(pred):
            return "🟢 Likely Not Autistic" if pred == 0 else "🔴 Likely Autistic"

        st.write(f"**Logistic Regression:** {label(pred_log)}")
        st.write(f"**SVM:** {label(pred_svm)}")
        st.write(f"**Random Forest:** {label(pred_rf)}")
        st.write(f"**Neural Network:** {label(pred_nn)}")

if __name__ == '__main__':
    main()
