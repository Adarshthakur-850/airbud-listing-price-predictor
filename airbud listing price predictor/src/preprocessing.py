import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocesses the dataframe: cleans missing values, encodes categoricals, scales features.
    Returns X (features) and y (target), and the encoders/scalers for future inference.
    """
    df = df.copy()
    
    # 1. Handle missing values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    # 2. Drop irrelevant columns for prediction
    drop_cols = ['id']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # 3. Encode Categorical Variables
    le = LabelEncoder()
    if 'room_type' in df.columns:
        df['room_type'] = le.fit_transform(df['room_type'])
    
    # 4. Separate Features and Target
    target_col = 'price'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        # For inference where target might not exist
        X = df
        y = None
        
    # 5. Scaling (Optional for Random Forest but good practice)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y, le, scaler
