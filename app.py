import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from keras.models import load_model

# Load the models
decision_tree_model = joblib.load('decisionTree_model.pkl')
logistic_regression_model = joblib.load('logReg_model.pkl')
svm_model = joblib.load('SVM_model.pkl')
neural_network_model = load_model('FFNN_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the column names used during training
column_names = ['distance_from_shore', 'distance_from_port', 'speed', 'course', 'lat', 'lon']

# Get user inputs
inputs = []
for column in column_names:
    value = input(f"Enter the value for {column}: ")
    inputs.append(float(value))

# Create a dataframe with the user inputs
df = pd.DataFrame([inputs], columns=column_names)

# Normalize the user inputs using the scaler
normalized_inputs = scaler.transform(df)

# Make predictions using the models
decision_tree_prediction = decision_tree_model.predict(normalized_inputs)[0]
logistic_regression_prediction = logistic_regression_model.predict(normalized_inputs)[0]
svm_prediction = svm_model.predict(normalized_inputs)[0]
neural_network_prediction = neural_network_model.predict(normalized_inputs)[0]
neural_network_prediction = np.argmax(neural_network_prediction)

# Convert the predictions to human-readable labels
prediction_labels = {0: 'Not Fishing', 1: 'Fishing'}

# Print the predictions
print("Decision Tree Prediction:", prediction_labels[decision_tree_prediction])
print("Logistic Regression Prediction:", prediction_labels[logistic_regression_prediction])
print("SVM Prediction:", prediction_labels[svm_prediction])
print("Neural Network Prediction:", prediction_labels[neural_network_prediction])
