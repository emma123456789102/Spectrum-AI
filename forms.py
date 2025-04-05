forms.py

from django import forms

class QuestionnaireForm(forms.Form):
    communication = forms.ChoiceField(choices=[('Verbal speech', 'Verbal speech'), ('Non-verbal', 'Non-verbal')])
    developmental = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    eye_contact = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    interest_in_children = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    pretend_play = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    repetitive_movements = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    response_to_name = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    routine_changes = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    sensory_sensitivities = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    social_cues = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    specific_interests = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
