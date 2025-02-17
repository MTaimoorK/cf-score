# Carbon Footprint Prediction API  

This is a Flask-based API that predicts an individual's carbon footprint based on various input parameters. It also provides SHAP values to explain the contribution of different factors to the prediction.  

## Features  
- Accepts JSON input with relevant carbon footprint data.  
- Uses a pre-trained model to predict the total carbon footprint.  
- Computes a weighted carbon footprint based on predefined weights.  
- Encodes categorical variables dynamically.  
- Provides SHAP values to explain the prediction.  

## Requirements  

Ensure you have Python installed along with the necessary dependencies. You can install them using:  

```bash
pip install flask pandas numpy joblib shap
```

## Setup  

1. Clone the repository or copy the files to your local machine.  
2. Place the trained model, encoder, and SHAP explainer in the `model/` directory:
   - `model.pkl` (pre-trained model)  
   - `encoder.pkl` (fitted encoders)  
   - `explainer.pkl` (SHAP explainer)  
3. Run the Flask app:  

```bash
python app.py
```

## API Usage  

### Endpoint: `/predict`  
- **Method:** `POST`  
- **Content-Type:** `application/json`  
- **Request Body Example:**  

```json
{
    "Grade": "A",
    "Gender": "Male",
    "Hosteler/Daily-Scholar": "Daily-Scholar",
    "Department_": "Computer Science",
    "CF_Washing": 3,
    "CF_for_Iron": 2,
    "CF_for_Meat": 5,
    "CF_of_Rice": 4,
    "CF_of_Vegetables": 3,
    "CF_of_Milk": 4,
    "CF_of_Fruit": 3,
    "CF_of_Air_Conditioner": 6,
    "CF_of_Hair_Dryer": 2,
    "CF_of_Laptop": 5,
    "CF_of_Paper": 3,
    "Total_CF_For_Transportation": 7
}
```

### Response Example  

```json
{
    "prediction": 23.5,
    "shap_values": [0.12, -0.05, 0.32, ...]
}
```

## Notes  
- Ensure the `model/` directory contains the necessary files before running the API.  
- If categorical variables contain unseen values, the encoder dynamically updates its classes.  
- The API provides SHAP values for better explainability of predictions.  
