from flask import Flask, render_template, request, jsonify,Blueprint
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

app3 = Blueprint('app3', __name__, template_folder='templates')

# Load the dataset
df = pd.read_csv("training.csv")  # Ensure the correct path

# Get the unique diseases from the prognosis column
unique_diseases = df['prognosis'].unique()

# Encode the prognosis (diseases) to numerical values
label_encoder = LabelEncoder()
df['prognosis_encoded'] = label_encoder.fit_transform(df['prognosis'])

# Use the columns from the dataset, excluding the 'prognosis' column
symptoms_list = df.columns[:-2].tolist()  # Exclude both 'prognosis' and 'prognosis_encoded'

# Prepare data for ML models
X = df[symptoms_list]
y = df['prognosis_encoded']

# Initialize the models
decision_tree = DecisionTreeClassifier().fit(X, y)
random_forest = RandomForestClassifier().fit(X, y)
naive_bayes = GaussianNB().fit(X, y)
knn = KNeighborsClassifier().fit(X, y)

# Function to predict diseases based on input symptoms
def predict_disease(selected_symptoms):
    input_data = np.zeros(len(symptoms_list))
    for symptom in selected_symptoms:
        if symptom in symptoms_list:
            input_data[symptoms_list.index(symptom)] = 1

    input_data = input_data.reshape(1, -1)

    # Get predictions from each model
    dt_prediction = decision_tree.predict(input_data)[0]
    rf_prediction = random_forest.predict(input_data)[0]
    nb_prediction = naive_bayes.predict(input_data)[0]
    knn_prediction = knn.predict(input_data)[0]

    # Decode the numerical predictions back to the disease names
    decoded_predictions = {
        "DecisionTree": label_encoder.inverse_transform([dt_prediction])[0],
        "RandomForest": label_encoder.inverse_transform([rf_prediction])[0],
        "NaiveBayes": label_encoder.inverse_transform([nb_prediction])[0],
        "KNN": label_encoder.inverse_transform([knn_prediction])[0]
    }

    return decoded_predictions

# Route for the home page (form to select symptoms)
@app3.route('/')
def home():
    return render_template('index3.html', symptoms=symptoms_list)

# Route to handle predictions
@app3.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    # Get the prediction results
    predictions = predict_disease(selected_symptoms)

    return jsonify(predictions)

if __name__ == '__main__':
    app3.run(debug=True)
