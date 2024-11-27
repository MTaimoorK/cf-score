from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap

# Load pre-trained model, scaler, and encoders
model = joblib.load('..model/model.pkl')
encoders = joblib.load('..model/encoder.pkl')
explainer = joblib.load('..model/explainer.pkl')

# Define weights for the columns
weights = {
        'CF_Washing': 1.0, 'CF_for_Iron': 0.5, 'CF_for_Meat': 1.5, 'CF_of_Rice': 1.2, 
        'CF_of_Vegetables': 1.0, 'CF_of_Milk': 1.3, 'CF_of_Fruit': 1.0, 
        'CF_of_Air_Conditioner': 2.0, 'CF_of_Hair_Dryer': 1.0, 'CF_of_Laptop': 1.5, 
        'CF_of_Paper': 0.8, 'Total_CF_For_Transportation': 2.5
}

# Initialize Flask app
app = Flask(__name__)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
        try:
            # Collect JSON input
            input_data = request.get_json()

            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Check for missing columns and handle them
            required_columns = ['Grade', 'Gender', 'Hosteler/Daily-Scholar', 'Department_', 
                                'CF_Washing', 'CF_for_Iron', 'CF_for_Meat', 'CF_of_Rice', 
                                'CF_of_Vegetables', 'CF_of_Milk', 'CF_of_Fruit', 
                                'CF_of_Air_Conditioner', 'CF_of_Hair_Dryer', 'CF_of_Laptop', 'CF_of_Paper', 'Total_CF_For_Transportation']

            missing_columns = [col for col in required_columns if col not in input_df.columns]
            if missing_columns:
                return jsonify({'error': f"Missing required columns: {missing_columns}"})

            # Add missing columns if necessary
            if 'Total_CF' not in input_df.columns:
                input_df['Total_CF'] = input_df[['CF_Washing', 'CF_for_Iron', 'CF_for_Meat', 'CF_of_Rice', 
                                                  'CF_of_Vegetables', 'CF_of_Milk', 'CF_of_Fruit', 
                                                  'CF_of_Air_Conditioner', 'CF_of_Hair_Dryer', 
                                                  'CF_of_Laptop', 'CF_of_Paper', 'Total_CF_For_Transportation']].sum(axis=1)

            # Calculate Weighted_CF based on the provided weights
            input_df['Weighted_CF'] = input_df[list(weights.keys())].mul(pd.Series(weights), axis=1).sum(axis=1)

            # Encode categorical variables
            categorical_columns = ['Grade', 'Gender', 'Hosteler/Daily-Scholar', 'Department_']
            for col in categorical_columns:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
                    # Handle unseen labels dynamically (if any)
                    if not set(input_df[col]).issubset(set(encoders.classes_)):
                        unique_values = np.unique(np.concatenate([encoders.classes_, input_df[col].unique()]))
                        # Dynamically update the encoder classes
                        encoders.classes_ = unique_values
                    input_df[col] = encoders.transform(input_df[col])

            # Ensure the input data has exactly the required columns in the correct order
            input_df = input_df[required_columns]

            # Reorder columns to match model's expected input
            input_data_for_prediction = input_df[model.feature_names_in_]

            # Prediction
            prediction = model.predict(input_data_for_prediction)

            # SHAP Explanation
            shap_values = explainer.shap_values(input_df)
        
            # Identify top features based on SHAP values
            top_features = {
                'features': input_df.columns.tolist(),
                'shap_values': shap_values[0].tolist()
            }

            # Return response
            return jsonify({
                'prediction': round(prediction[0], 2),
                'top_features': top_features,
                'input_data': input_df.to_dict(),
                'shap_values': shap_values[0].tolist(),
            })

        except Exception as e:
            return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
        app.run(debug=True)