import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle as pkl

# Load dataset
data = pd.read_csv("autism_screening.csv")

# Preprocessing
data = data.drop(columns=['ID']) if 'ID'in data.columns else data
data = data.dropna()

# Display dataset information
print("Dataset Overview:")
print(data.head())
print(data.info())

# Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(data[col]) for col in categorical_columns}
for col, le in label_encoders.items():
    data[col] = le.transform(data[col])

# Define features and target
X = data.drop(columns=['A1_Score'])
y = data['A1_Score']
X, y = X.apply(pd.to_numeric, errors='coerce').dropna(), pd.to_numeric(y, errors='coerce')[X.index]

# Handle class imbalance using SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Standardize features
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

# Models and Training
models = {
    "Random Forest": GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                   {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30],
                                    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                                   cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                     max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression()
}

# Train and Evaluate Models
model_results = {}
plt.figure(figsize=(8, 6))
for name, model in models.items():
    print(f"\nTraining {name}...")
    if isinstance(model, GridSearchCV):
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
    else:
        model.fit(X_train_scaled if name != "Random Forest" else X_train, y_train)
        best_model = model

    y_pred = best_model.predict(X_test_scaled if name != "Random Forest" else X_test)
    y_probs = best_model.predict_proba(X_test_scaled if name != "Random Forest" else X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
# --- Visualisations for the comparision and the findings chapter  ----
    model_results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred)
    }
    print(f"{name} Accuracy: {model_results[name]['Accuracy']}")
    print(f"{name} ROC-AUC: {model_results[name]['ROC-AUC']}")
    print(model_results[name]['Classification Report'])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()



# Visualization: ROC Curve Comparison
def plot_roc_comparison(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.show()


# ROC Curve Visualization
def plot_roc_curve(y_test, model, X_test, label):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.2f}')

plt.figure(figsize=(8, 6))
plot_roc_curve(y_test, rf_best, X_test, "Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Determine Best Model
best_model_name = max(model_results, key=lambda k: model_results[k]["ROC-AUC"])
print(f"\nBest Model: {best_model_name} with ROC-AUC: {model_results[best_model_name]['ROC-AUC']}")

# Save Best Model
print("\nSaving the best model...",best_model_name)
joblib.dump(models[best_model_name], 'autism_best_model.pkl')

