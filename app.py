from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, origins='*')

# Load the serialized model
try:
    with open('kidney_prediction_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")


# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Convert data to DataFrame
        input_data = pd.DataFrame(data, index=[0])

        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)

        # Return the prediction as JSON response
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print(f"Error predicting: {e}")
        return jsonify({'error': 'An error occurred while predicting.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
