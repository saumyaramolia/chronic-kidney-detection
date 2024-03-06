from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, origins='*')

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load the serialized model
try:
    with open('kidney_prediction_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    loaded_model = None

# Map prediction labels to human-readable output
prediction_labels = {0: "Not a patient of CKD", 1: "Patient has CKD"}


# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if loaded_model is None:
            return jsonify({'error': 'Model not loaded.'}), 500

        # Get data from the request
        data = request.get_json()

        # Convert data to DataFrame
        input_data = pd.DataFrame(data, index=[0])

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)

        # Convert prediction to human-readable output
        prediction_output = prediction_labels.get(prediction[0], "Unknown")

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction_output})
    except Exception as e:
        logging.error(f"Error predicting: {e}")
        return jsonify({'error': 'An error occurred while predicting.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
