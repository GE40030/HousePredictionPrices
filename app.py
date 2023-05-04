import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor


# Initialize Flask application
app = Flask(__name__)

# Load the trained model
with open('gb_prediction_model.pkl', 'rb') as file:
    gb_prediction_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')    

# Define endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    # Convert data into numpy array
    X = np.array(data['features'])
    # Make prediction using loaded model
    prediction = gb_prediction_model.predict(X)
    # Return prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

# Run the application
if __name__ == '__main__':
    app.run(debug=True)

