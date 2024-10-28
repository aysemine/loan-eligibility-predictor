from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load('loan_status_model.pkl')

# Define the base and risk adjustment interest rates
base_interest_rate = 0.11011694892245036
risk_adjustment_factor = 0.11011694892245036

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

from flask import Flask, render_template, request

# ... your existing code ...

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form

    # Convert input data into a DataFrame
    input_data = pd.DataFrame({
        'person_age': [int(data['person_age'])],
        'person_income': [int(data['person_income'])],
        'personhomeownership': [data['personhomeownership']],
        'personemplength': [int(data['personemplength'])],
        'loan_intent': [data['loan_intent']],
        'loan_grade': [data['loan_grade']],
        'loan_amnt': [int(data['loan_amnt'])],
        'loanpercentincome': [float(data['loanpercentincome'])],
        'cbpersondefaultonfile': [data['cbpersondefaultonfile']],
        'cbpresoncredhistlength': [int(data['cbpresoncredhistlength'])]
    })

    # One-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align input data with the training data
    X_test = pd.DataFrame(columns=model.feature_names_in_)
    input_data = input_data.reindex(columns=X_test.columns, fill_value=0)  # Reindex to match the model's input
    X_test = pd.concat([X_test, input_data], ignore_index=True)

    # Make predictions
    loan_status_prob = model.predict_proba(X_test)[0][1]  # Probability of default

    # Determine eligibility and calculate the interest rate if eligible
    if loan_status_prob > 0.5:
        return render_template('result.html', message="Customer can't get credit due to high default risk.", new_interest_rate=None, predicted_default_probability=None)
    else:
        new_interest_rate = base_interest_rate + (risk_adjustment_factor * loan_status_prob)
        return render_template(
            'result.html',
            message="Customer is eligible for credit.",
            new_interest_rate=new_interest_rate,
            predicted_default_probability=loan_status_prob
        )


if __name__ == "__main__":
    app.run(debug=True)
