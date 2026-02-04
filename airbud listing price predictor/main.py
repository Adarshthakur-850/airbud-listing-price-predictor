import argparse
import sys
import pandas as pd
from src.train import train_pipeline
from src.predict import predict_price

def main():
    parser = argparse.ArgumentParser(description="Airbnb Listing Price Predictor")
    parser.add_argument('--train', action='store_true', help='Run the training pipeline')
    parser.add_argument('--predict', action='store_true', help='Run inference on example data')
    
    args = parser.parse_args()
    
    if args.train:
        train_pipeline()
        
    elif args.predict:
        # Example data for prediction
        example_data = [
            {
                'latitude': 40.7,
                'longitude': -74.0,
                'room_type': 'Entire home/apt',
                'minimum_nights': 3,
                'number_of_reviews': 50,
                'reviews_per_month': 2.5,
                'calculated_host_listings_count': 2,
                'availability_365': 180
            },
            {
                'latitude': 40.6,
                'longitude': -73.9,
                'room_type': 'Private room',
                'minimum_nights': 1,
                'number_of_reviews': 10,
                'reviews_per_month': 0.5,
                'calculated_host_listings_count': 1,
                'availability_365': 300
            }
        ]
        
        print(f"Predicting for example data:\n{pd.DataFrame(example_data)}")
        try:
            predictions = predict_price(example_data)
            print("\nPredictions:")
            for i, pred in enumerate(predictions):
                print(f"Item {i+1}: ${pred:.2f}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
