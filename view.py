# autismapp/views.py

from django.shortcuts import render
from .forms import QuestionnaireForm
import numpy as np
import joblib

model = joblib.load('path/to/your/model.pkl')  # load the trained model once

def questionnaire_view(request):
    if request.method == 'POST':
        form = QuestionnaireForm(request.POST)
        if form.is_valid():
            answers = form.cleaned_data

            yes_no_map = {"Yes": 1, "No": 0}
            communication_map = {"Verbal speech": 1, "Non-verbal": 0}

            features = [
                communication_map.get(answers['communication'], 0),
                yes_no_map.get(answers['developmental'], 0),
                yes_no_map.get(answers['eye_contact'], 0),
                yes_no_map.get(answers['interest_in_children'], 0),
                yes_no_map.get(answers['pretend_play'], 0),
                yes_no_map.get(answers['repetitive_movements'], 0),
                yes_no_map.get(answers['response_to_name'], 0),
                yes_no_map.get(answers['routine_changes'], 0),
                yes_no_map.get(answers['sensory_sensitivities'], 0),
                yes_no_map.get(answers['social_cues'], 0),
                yes_no_map.get(answers['specific_interests'], 0),
            ]

            input_data = np.array(features).reshape(1, -1)

            prediction = model.predict(input_data)
            diagnosis = "Likely autistic" if prediction[0] == 1 else "Not autistic"

            return render(request, 'result.html', {'diagnosis': diagnosis})
    else:
        form = QuestionnaireForm()

    return render(request, 'questionnaire.html', {'form': form})
