from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



app = Flask(__name__)
diabetes_model = pickle.load(open('diabetes.pkl', 'rb'))
# malaria = pickle.load(open('malaria.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
# covid = pickle.load(open('covid.pkl', 'rb'))

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/diabetes", methods=["GET"])
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def breast():
    return render_template("heart.html")

@app.route("/melanoma")
def covid():
    return render_template("melanoma.html")

@app.route("/breast-cancer")
def malaria():
    return render_template("breast.html")


@app.route("/diabetesresult", methods=["POST"])
def diabetes_result():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    my_prediction = diabetes_model.predict(features_value)
    if my_prediction > 0.5:
        result = "There's a Pretty high chance you have Diabetes! Please consult a doctor ASAP!"
        message = "Emergency!"
    else:
        result = "You're safe! There's a low chance you have diabetes. But don't start binging on those chocolates!"
        message = "Congratulations!"

    return render_template("diabetes-result.html", prediction = result, alert = message)


@app.route("/heartresult", methods=["POST"])
def heart_result():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    my_prediction = heart_model.predict(features_value)
    if my_prediction > 0.5:
        result = "There's a Pretty high chance you have a Heart Disease! Please consult a doctor ASAP!"
        message = "Emergency!"
    else:
        result = "You're safe! There's a low chance you have a heart disease. But it's still good to stay away from oily food!"
        message = "Congratulations!"

    return render_template("heart-result.html", prediction = result, alert = message=)


if __name__ == ("__main__"):
    app.run(debug=True)