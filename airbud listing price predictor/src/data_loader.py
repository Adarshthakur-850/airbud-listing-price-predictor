import pandas as pd
import numpy as np
import os

def load_data(filepath="data/listings.csv"):
    """
    Loads data from CSV. If file doesn't exist, generates synthetic data.
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Generating synthetic data...")
        generate_dummy_data(filepath)
    
    return pd.read_csv(filepath)

def generate_dummy_data(filepath):
    """
    Generates synthetic Airbnb-like data for demonstration.
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(1, n_samples + 1),
        'latitude': np.random.uniform(40.5, 40.9, n_samples),
        'longitude': np.random.uniform(-74.2, -73.7, n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples, p=[0.6, 0.35, 0.05]),
        'minimum_nights': np.random.randint(1, 15, n_samples),
        'number_of_reviews': np.random.randint(0, 300, n_samples),
        'reviews_per_month': np.random.uniform(0, 10, n_samples),
        'calculated_host_listings_count': np.random.randint(1, 10, n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'price': [] # To be generated based on features
    }
    
    # Generate price with some logic so the model has something to learn
    base_price = 50
    for i in range(n_samples):
        price = base_price
        
        # Room type factor
        if data['room_type'][i] == 'Entire home/apt':
            price += 100
        elif data['room_type'][i] == 'Private room':
            price += 40
            
        # Review factor (popular places might be slightly more expensive or cheaper, let's say popularity adds value)
        price += data['number_of_reviews'][i] * 0.1
        
        # Availability factor (scarce availability might imply high demand)
        if data['availability_365'][i] < 50:
            price += 20
            
        # Random noise
        price += np.random.normal(0, 20)
        
        data['price'].append(max(10, int(price))) # Ensure price is positive
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Synthetic data saved to {filepath}")
