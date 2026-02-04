import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import build_model

def train_pipeline(data_path="data/listings.csv", model_dir="models"):
    """
    Orchestrates the training pipeline: load -> preprocess -> train -> eval -> save.
    """
    print("Starting Training Pipeline...")
    
    # 1. Load Data
    df = load_data(data_path)
    print(f"Data Loaded: {df.shape}")
    
    # 2. Preprocess Data
    X, y, le, scaler = preprocess_data(df)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Build Model
    model = build_model()
    
    # 5. Train Model
    print("Training Random Forest Model...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print("Model Evaluation:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.2f}")
    print("-" * 30)
    
    # 7. Save Artifacts
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "price_model.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model and artifacts saved to {model_dir}")
