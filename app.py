import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Big Mart Sales Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    
    st.write("### Data Preview")
    st.write(df.head())
    
    # Display columns
    st.write("Columns in dataset:", df.columns.tolist())
    
    # Select target variable
    target = st.selectbox("Select target variable", df.columns)
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target])
    
    # Select features excluding the target column
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]
    
    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Ensure X contains only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Handle missing values only for numerical columns
    X = X.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)
    
    # Ensure y is numeric and handle missing/infinite values
    y = pd.to_numeric(y, errors='coerce')  # Convert non-numeric to NaN
    y.fillna(y.mean(), inplace=True)  # Fill NaN with mean
    y = np.array(y, dtype=np.float64)  # Convert to NumPy array
    y = np.nan_to_num(y)  # Replace any remaining NaNs/Infs with 0
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Debugging shapes and types
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    st.write(f"X_train type: {type(X_train)}, y_train type: {type(y_train)}")
    
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in X_train and y_train sizes!"
    assert np.isfinite(X_train).all().all(), "X_train contains infinite or NaN values!"
    assert np.isfinite(y_train).all(), "y_train contains infinite or NaN values!"
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    st.write("### Model Performance")
    st.write(f"Mean Absolute Error: {mae:.2f}")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Save the model
    with open("big_mart_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    st.success("Model trained and saved successfully!")
    
    # Prediction section
    st.write("### Make Predictions")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))
    
    if st.button("Predict Sales"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.write(f"### Predicted Sales: {prediction:.2f}")
