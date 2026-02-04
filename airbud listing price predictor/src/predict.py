import joblib
import pandas as pd
import numpy as np
import os

def predict_price(input_data):
    """
    Loads model artifacts and predicts price for input data.
    input_data: list of dicts or DataFrame containing feature columns.
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "price_model.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, le_path, scaler_path]):
        raise FileNotFoundError("Model artifacts not found. Please train the model first.")
        
    # Load artifacts
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare input
    if isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = input_data.copy()
        
    # Preprocessing for Inference
    # 1. Fill NA
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    # 2. Encode
    # Handle unseen labels by mapping them to a default (e.g., mode) or raising error.
    # Here we'll try to transform, if fail, we might need robust handling.
    try:
        df['room_type'] = le.transform(df['room_type'])
    except ValueError as e:
        print(f"Warning: Unseen label in room_type. {e}")
        # Fallback: assign to most frequent class (0 usually) or similar
        # For simplicity in this demo, we assume valid input or let it crash
        
    # 3. Drop ID if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # 4. Scale
    # Ensure columns match scaler's expected input
    # Reorder columns to match training order (important!)
    # We need to know the columns scaler was trained on. 
    # Ideally, we should save feature names, but here we assume correct schema from user.
    
    X_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(X_scaled)
    return prediction
