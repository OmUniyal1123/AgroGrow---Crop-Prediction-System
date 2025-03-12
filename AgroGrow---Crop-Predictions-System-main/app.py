import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

# Load the model using one of the solutions
model = pickle.load(open(r"C:\Users\omuni\Downloads\AgroGrow---Crop-Predictions-System-main\AgroGrow---Crop-Predictions-System-main\model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)