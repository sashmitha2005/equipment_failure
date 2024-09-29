from flask import Flask, request, jsonify, render_template  # Import render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})  # Enable CORS for the React frontend

# Load dataset
df = pd.read_csv('equipment_failure_dataset.csv')

# Preprocessing steps
df = df.drop(columns=['EquipmentID'])  # Drop EquipmentID as it's not useful for prediction
X = df.drop(columns=['Failure'])
y = df['Failure'].apply(lambda x: 1 if x == 'Yes' else 0)

categorical_cols = ['Location', 'Environment']
numerical_cols = ['Age', 'UsageHours', 'MaintenanceHistory', 'Temperature', 'Pressure', 'VibrationLevel', 'OperatorExperience', 'FailureHistory']

# Create preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models for Voting Classifier
models = [
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('lr', LogisticRegression(max_iter=1000)),
    ('svc', SVC(probability=True)),
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier()),
    ('xgb', XGBClassifier())
]

# Voting Classifier
voting_clf = VotingClassifier(estimators=models, voting='soft')

# Create pipeline with preprocessor and model
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

# Train the model within the pipeline
clf_pipeline.fit(X_train, y_train)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Validate input data
    try:
        features = [
            float(data['age']),
            float(data['usageHours']),
            float(data['maintenanceHistory']),
            float(data['temperature']),
            float(data['pressure']),
            float(data['vibrationLevel']),
            float(data['operatorExperience']),
            float(data['failureHistory']),
            data['location'],
            data['environment']
        ]
    except (KeyError, ValueError) as e:
        logging.error(f'Input error: {e}')
        return jsonify({'error': 'Invalid input data'}), 400

    # Create a DataFrame for the features
    feature_names = numerical_cols + categorical_cols  # Combine numerical and categorical column names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Make prediction using the trained model
    try:
        prediction = clf_pipeline.predict(features_df)
    except Exception as e:
        logging.error(f'Model prediction error: {e}')
        return jsonify({'error': 'Prediction failed'}), 500

    # Convert the prediction result to 'Equipment fails' or 'Equipment not fails'
    result = 'Equipment fails' if prediction[0] == 1 else 'Equipment not fails'

    # Return the result as plain text
    return result, 200  # Returning plain text instead of JSON

# Route to render the HTML template
@app.route('/')
def home():
    return render_template('index.html')  # Renders the index.html file

if __name__ == '__main__':
    app.run(debug=True, port=5000)
