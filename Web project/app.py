from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

# Read the symptoms from symptoms.txt (they are comma separated) and store them in a list
with app.open_resource('static/symptoms.txt') as f:
    symptoms = f.read().decode('utf-8').split(',')

# Create a new list with the symptoms capitalized and without the underscore
readable_symptoms = [symptom.replace('_', ' ').capitalize() for symptom in symptoms]

# load the models
with app.open_resource('../Models/knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)
with app.open_resource('../Models/rfc_model.pkl', 'rb') as f:
    rfc = pickle.load(f)
with app.open_resource('../Models/svc_model.pkl', 'rb') as f:
    svc = pickle.load(f)
with app.open_resource('../Models/lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)
with app.open_resource('../Models/nn_model.pkl', 'rb') as f:
    mlp = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html', symptoms=zip(symptoms, readable_symptoms))
 
@app.route("/predict",methods=["POST","GET"])
def predict():
   
    if request.method == 'POST':
        current_symptoms = request.form['hidden_symptoms']
        print(current_symptoms)     
        disease = predict_disease(current_symptoms)
    return render_template('results.html', disease=disease[0], result=disease[1])


def predict_disease(current_symptoms):
    # We will try to predict the disease using all the models and then return the best option
    knn_prediction = predict_disease_by_model(current_symptoms, knn)
    rfc_prediction = predict_disease_by_model(current_symptoms, rfc)
    svc_prediction = predict_disease_by_model(current_symptoms, svc)
    lr_prediction = predict_disease_by_model(current_symptoms, lr)
    mlp_prediction = predict_disease_by_model(current_symptoms, mlp)

    # First we will get the disease with the highest votes
    predictions = [knn_prediction, rfc_prediction, svc_prediction, lr_prediction, mlp_prediction]

    return [get_best_prediction(predictions), predictions]


def predict_disease_by_model(current_symptoms, model):

    current_symptoms = current_symptoms.split(',')
    # we need to create a dataframe with the same columns as the training dataset
    symptoms_df = pd.DataFrame(columns=symptoms)
    # we need to add the symptoms to the dataframe
    symptoms_df.loc[0] = 0
    for symptom in current_symptoms:
        symptoms_df[symptom] = 1
    # predict the disease using both models
    result = []
    # the result should be a list containing the disease and the probability
    disease = model.predict(symptoms_df)
    probability = np.max(model.predict_proba(symptoms_df))
    result.append(disease[0])
    result.append(probability)
    # add the name of the model to the result
    result.append(model.__class__.__name__)

    return result
    

def get_best_prediction(predictions):
    # Initialize a dictionary to count the number of votes for each disease
    vote_counts = {}
    for prediction in predictions:
        disease = prediction[0]
        vote_counts[disease] = vote_counts.get(disease, 0) + 1

    print(predictions)
    print(vote_counts)

    # Find the disease(s) with the most votes
    max_votes = max(vote_counts.values())
    top_diseases = [disease for disease, votes in vote_counts.items() if votes == max_votes]
    
    # If there is only one top disease, return it
    if len(top_diseases) == 1:
        return top_diseases[0]
    else:
        # Otherwise, find the top disease(s) with the highest probability
        top_proba = 0
        top_disease = None
        for disease, proba in predictions:
            if disease in top_diseases and proba > top_proba:
                top_proba = proba
                top_disease = disease
        return top_disease

    
 
if __name__ == "__main__":
    app.run(debug=True)