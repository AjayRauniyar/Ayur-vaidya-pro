from flask import Flask, request, jsonify, render_template,Blueprint

app2 = Blueprint('app2', __name__, template_folder='templates')

# Function to match symptoms to diseases and return recommendations
def match_symptom(symptom):
    known_symptoms = ["Fever", "Coughing", "Weakness", "Body ache", "Chest pain"]
    for known in known_symptoms:
        if known.lower() in symptom.lower():
            return known
    return None

def get_ingredients_from_matching_diseases(symptoms, threshold=0.5):
    disease_data = {
        "Jwara": {
            "symptoms": ["Fever", "Thirst", "Weakness", "Body ache", "Loss of appetite"],
            "ingredients": ["Guduchi", "Tulsi", "Dry Ginger", "Pippali", "Coriander seeds"]
        },
        "Kasa": {
            "symptoms": ["Coughing", "Wheezing", "Chest pain", "Difficulty in breathing"],
            "ingredients": ["Vasa", "Licorice", "Pippali", "Honey", "Black Pepper"]
        }
    }
    
    matched_diseases = []
    matched_ingredients = set()
    
    for disease, info in disease_data.items():
        matched_count = sum(1 for symptom in symptoms if symptom in info["symptoms"])
        if (matched_count / len(info["symptoms"])) >= threshold:
            matched_diseases.append(disease)
            matched_ingredients.update(info["ingredients"])
    
    return matched_diseases, matched_ingredients

def predict_and_verify(symptoms):
    matched_symptoms = []
    for symptom in symptoms:
        match = match_symptom(symptom)
        if match:
            matched_symptoms.append(match)
    
    matched_diseases, matched_ingredients = get_ingredients_from_matching_diseases(matched_symptoms)
    
    if matched_diseases:
        return {
            "diseases": matched_diseases,
            "ingredients": list(matched_ingredients)
        }
    else:
        return {
            "message": "No matching diseases found based on the provided symptoms.",
            "ingredients": ["Guduchi", "Tulsi", "Dry Ginger", "Honey", "Black Pepper"]
        }

@app2.route('/')
def home2():
    return render_template('index2.html')

@app2.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '')
    user_symptoms = [symptom.strip() for symptom in symptoms.split(',')]
    result = predict_and_verify(user_symptoms)
    return jsonify(result)

if __name__ == '__main__':
    app2.run(debug=True)
