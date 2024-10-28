# Loan Eligibility Predictor

This project is a Flask web application that predicts loan eligibility based on user inputs. It uses a pre-trained machine learning model to assess the likelihood of a customer defaulting on a loan, and, if eligible, provides a suggested interest rate based on the customer's risk profile.

## Features

- **User Input Form**: A user-friendly interface for inputting customer data, including age, income, home ownership, employment length, loan intent, loan grade, and credit history length.
- **Loan Eligibility Prediction**: Predicts loan eligibility based on a trained model.
- **Interest Rate Suggestion**: Calculates a risk-adjusted interest rate if the customer is eligible.
- **Default Probability Estimation**: Shows the likelihood of default as a percentage for additional insight.

## Tech Stack

- **Frontend**: HTML, CSS (Bootstrap for styling)
- **Backend**: Flask
- **Machine Learning**: Pre-trained model using `joblib`

## Installation

### Prerequisites

- Python
- Flask
- joblib
- Pandas
- Numpy

