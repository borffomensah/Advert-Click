import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the saved Decision Tree model
def load_model():
    with open('dt.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define the mapping dictionary for categorical features
label_encoders = {
    'gender': {'Female': 0, 'Male': 1, 'Non-Binary': 2},
    'device_type': {'Desktop': 0, 'Mobile': 1, 'Tablet': 2},
    'ad_position': {'Top': 2, 'Side': 1, 'Bottom': 0},
    'browsing_history': {
        'Shopping': 3, 'Entertainment': 1, 'Education': 0, 'Social Media': 4, 'News': 2
    },
    'time_of_day': {'Afternoon': 0, 'Morning': 2, 'Night': 3, 'Evening': 1}
}

# Function to preprocess input data
def preprocess_input(data):
    for column, encoder in label_encoders.items():
        if column in data:
            data[column] = encoder.get(data[column], -1)  # Default to -1 if value not found
    return data

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Ad Click Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Preprocess input
        processed_data = preprocess_input(data)

        # Convert to DataFrame
        input_df = pd.DataFrame([processed_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Format response
        result = {
            "prediction": int(prediction),
            "message": "User is likely to click on the ad." if prediction == 1 else "User is unlikely to click on the ad."
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
