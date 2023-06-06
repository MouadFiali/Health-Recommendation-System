from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import ast
import random
import pandas as pd
import chardet
import numpy as np


app = Flask(__name__)

# Set the user info global variable
user_info = []

def load_collab_filtering_models():
    global collab_mlp, collab_nb, collab_pm, collab_sgd
    # load the models
    with app.open_resource('..\\Models\\collaborative models\\mlp_model.pkl', 'rb') as f:
        collab_mlp = pickle.load(f)
    with app.open_resource('..\\Models\\collaborative models\\nb_model.pkl', 'rb') as f:
        collab_nb = pickle.load(f)
    with app.open_resource('..\\Models\\collaborative models\\pm_model.pkl', 'rb') as f:
        collab_pm = pickle.load(f)
    with app.open_resource('..\\Models\\collaborative models\\sgd_model.pkl', 'rb') as f:
        collab_sgd = pickle.load(f)


def load_utility_filtering_models():
    global utility_nb, utility_mlp, utility_sgd
    # load the models
    with app.open_resource('../Models/utility models/bernoulliNB.pkl', 'rb') as f:
        utility_nb = pickle.load(f)
    with app.open_resource('../Models/utility models/mlp.pkl', 'rb') as f:
        utility_mlp = pickle.load(f)
    with app.open_resource('../Models/utility models/sgd.pkl', 'rb') as f:
        utility_sgd = pickle.load(f)

# load the models
load_collab_filtering_models()
load_utility_filtering_models()

# Read the symptoms from symptoms.txt (they are comma separated) and store them in a list
with app.open_resource('..\\Datasets\\Patient-disease-symptom\\collab_evidences.txt', 'r') as f:
    # the evidences are stored as a list in the file, so we need to convert it to a list
    collab_evidence_columns = ast.literal_eval(f.read())

with app.open_resource('..\\Datasets\\Patient-disease-symptom\\utility_evidences.txt', 'r') as f:
    # the evidences are stored as a list in the file, so we need to convert it to a list
    utility_evidence_columns = ast.literal_eval(f.read())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user', methods=['POST', 'GET'])
def get_user():
    if request.method == 'POST':
        user_info.clear()
        # set the user info (Name, age and sex)
        user_info.append(request.form.get('name'))
        user_info.append(request.form.get('age'))
        user_info.append(request.form.get('genre'))

        print(user_info)

        return redirect(url_for('main_page'))
    else:
        return render_template('index.html')
    
@app.route('/main')
def main_page():
    name = user_info[0]
    return render_template('main.html', name=name)
 
@app.route("/predict",methods=["POST","GET"])
def predict():
   
    json_response = ""

    if request.method == 'POST':

        data = request.get_json()  # Get JSON data from the request body
        prompt = data.get('prompt')

        print(prompt)     

        evidences_list = get_evidences(prompt)

        print(user_info)
        print(evidences_list)

        # predict the disease
        predictions = predict_disease(evidences_list)

        response = create_response(predictions[0], predictions[1], predictions[2])

        print("Response: ")
        print(response)

        json_response = {"response": response}

    return jsonify(json_response)


def create_response(disease, collab_prediction, utility_prediction):

    # get the unique predicted diseases
    unique_diseases = set()
    for prediction in collab_prediction:
        unique_diseases.add(prediction[0])
    for prediction in utility_prediction:
        unique_diseases.add(prediction[0])

    unique_diseases = list(unique_diseases) # to use after

    # create a response
    response = "La maladie la plus probable est: " + disease + ".\n\n"
    
    # add the collaborative predictions
    response += "Voici les maladies prédites par les modèles collaboratifs:\n"
    for prediction in collab_prediction:
        response += prediction[0] + " (score: " + str(prediction[1]) + ")\n"

    # add the utility predictions
    response += "\nVoici les maladies prédites par les modèles utilitaires:\n"
    for prediction in utility_prediction:
        response += prediction[0] + " (probabilité: " + str(prediction[1]) + ")\n"

    return response


def get_evidences(prompt):
    # TODO: get the symptoms from the prompt
    # for now let's use random symptoms
    prompt_evidences = [collab_evidence_columns[random.randint(0, len(collab_evidence_columns) - 1)] for i in range(random.randint(1, 20))] # TO BE CHANGED
    return prompt_evidences



def predict_disease(evidences):
    # We will try to predict the disease using all the models and then return the best option

    collaborative_predictions = []
    utility_predictions = []

    # predict the disease using the collaborative models
    for model in [collab_mlp, collab_nb, collab_pm, collab_sgd]:
        collaborative_predictions.append(collab_predict(evidences, model))

    # predict the disease using the utility models
    for model in [utility_nb, utility_mlp, utility_sgd]:
        utility_predictions.append(utility_predict(evidences, model))

    # get the best prediction
    best_prediction = get_best_prediction(collaborative_predictions, utility_predictions)

    return best_prediction


def utility_predict(evidences, model):

    columns = utility_evidence_columns
    # create a dataframe with 0 in all columns
    evidences_df = pd.DataFrame([[0 for i in range(len(columns))]], columns=columns)

    # process the evidences
    evidences_list = []
    for evidence in evidences:
        evidences_list.append(evidence.split('_@_')[0])

    print(evidences_list)

    # we need to add the evidences to the dataframe
    evidences_df.loc[0, evidences_list] = 1

    # create a list to store the result
    result = []

    # the result should be a list containing the disease and the probability and the name of the model
    result.append(model.predict(evidences_df)[0])
    result.append(model.predict_proba(evidences_df).max())
    result.append(model.__class__.__name__)

    return result

def collab_predict(evidences, model):

    # we need to create a dataframe with the same columns as the dataset
    columns = collab_evidence_columns
    evidences_df = pd.DataFrame([[0 for i in range(len(columns))]], columns=columns)

    # set the age and sex
    evidences_df.loc[0, 'AGE'] = int(user_info[1])
    evidences_df.loc[0, 'SEX'] = int(user_info[2])

    # we need to add the evidences to the dataframe
    evidences_df.loc[0, evidences] = 1
    
    # create a list to store the result
    result = []

    # the result should be a list containing the disease and the probability
    result.append(model.predict(evidences_df)[0])
    with app.open_resource('../Models/collaborative models/models_score.csv', 'r') as f:
        accuracy = pd.read_csv(f)
        # the accuracy dataframe contains lines with the name of the model and its score
        # we need to get the score of the model we are using
        score = accuracy[accuracy['Model'] == model.__class__.__name__]['Score'].values[0]
        result.append(score)

    # add the name of the model to the result
    result.append(model.__class__.__name__)

    return result
    

def get_best_prediction(collaborative_predictions, utility_predictions):
    # variable to store the best prediction
    best_prediction = ''

    # Initialize a dictionary to count the number of votes for each disease
    vote_counts = {}
    for prediction in collaborative_predictions:
        disease = prediction[0]
        vote_counts[disease] = vote_counts.get(disease, 0) + 1

    print(collaborative_predictions)
    print(vote_counts)

    # Find the disease(s) with the most votes
    max_votes = max(vote_counts.values())
    top_diseases = [disease for disease, votes in vote_counts.items() if votes == max_votes]
    
    # If there is only one top disease, return it
    if len(top_diseases) == 1:
        best_prediction = top_diseases[0]

    else:
        # In this case we need to use the utility models to find the best disease
        # Initialize a dictionary to count the number of votes for each disease
        utility_vote_counts = {}
        for prediction in utility_predictions:
            disease = prediction[0]
            utility_vote_counts[disease] = utility_vote_counts.get(disease, 0) + 1
        
        # Find the disease(s) with the most votes
        utility_max_votes = max(utility_vote_counts.values())
        utility_top_diseases = [disease for disease, votes in utility_vote_counts.items() if votes == utility_max_votes]

        # check for the common diseases between the two lists
        common_diseases = [disease for disease in top_diseases if disease in utility_top_diseases]

        # if there is only one common disease, return it
        if len(common_diseases) == 1:
            best_prediction = common_diseases[0]
        elif len(common_diseases) == 0:
            # if there are no common diseases, return the collaborative model's prediction with the highest accuracy
            max_accuracy = 0
            for prediction in collaborative_predictions:
                if prediction[1] > max_accuracy:
                    max_accuracy = prediction[1]
                    best_prediction = prediction[0]
        else:
            # if there are more than one common disease, pick the one with highest highest accuracy from the common diseases
            max_accuracy = 0
            for prediction in collaborative_predictions:
                if prediction[1] > max_accuracy and prediction[0] in common_diseases:
                    max_accuracy = prediction[1]
                    best_prediction = prediction[0]
    
    return [best_prediction, collaborative_predictions, utility_predictions]


    
if __name__ == "__main__":
    app.run(debug=True)