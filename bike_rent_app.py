import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# App Title
st.title("Bike Rental Demand Prediction")
st.write("""
This app allows you to:
- Upload a dataset.
- Train a Random Forest regression model.
- Evaluate model performance.
- Make predictions for bike rentals based on user input.
""")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV Dataset", type=["csv"])
if uploaded_file is not None:
    # Load Dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview:")
    st.write(data.head())

    # Data Cleaning
    st.write("### Data Cleaning:")
    data.replace('?', np.nan, inplace=True)
    numerical_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
    categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'workingday', 'weathersit']

    # Convert numerical columns to float
    for col in numerical_cols:
        data[col] = data[col].astype(float)

    # Impute missing values
    for col in numerical_cols:
        data[col].fillna(data[col].mean(), inplace=True)
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    st.write("Missing values handled successfully.")

    # Feature Engineering
    st.write("### Feature Engineering:")
    scaler = MinMaxScaler()
    numerical_scaled = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)
    categorical_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
    processed_data = pd.concat([numerical_scaled, categorical_encoded, data[['cnt']]], axis=1)

    # Train-Test Split
    X = processed_data.drop('cnt', axis=1)
    y = processed_data['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    st.write("### Model Training:")
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate Model
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    st.write(f"Model Performance:")
    st.write(f"  RMSE: {rmse:.2f}")
    st.write(f"  R²: {r2:.4f}")

    # Save the model
    joblib.dump(model, "random_forest_model.pkl")
    st.write("Model saved as 'random_forest_model.pkl'")

    # Prediction
    st.write("### Make Predictions:")
    st.write("Enter feature values to predict bike rentals.")
    feature_values = []
    for col in X.columns:
        value = st.number_input(f"Enter value for {col}:", min_value=0.0, max_value=1.0, step=0.01)
        feature_values.append(value)

    if st.button("Predict"):
        input_features = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(input_features)
        st.write(f"Predicted Bike Rentals: {int(prediction[0])}")
