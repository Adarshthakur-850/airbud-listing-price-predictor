# Airbud Listing Price Predictor (Airbnb Price Prediction)

This project predicts the listing price of Airbnb properties based on features like location, room type, and reviews. 
*(Note: "Airbud" in the directory name is assumed to be a typo for "Airbnb")*

## Project Structure
- `data/`: Contains the dataset (synthetic data generated if missing).
- `models/`: Stores trained model and preprocessing artifacts.
- `src/`: Source code for data loading, processing, model building, and inference.
- `main.py`: CLI entry point.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the model (and generate synthetic data if needed):
```bash
python main.py --train
```
This will save the model and artifacts to the `models/` directory.

### Prediction
To run inference on example data:
```bash
python main.py --predict
```

## Model
The project uses a **Random Forest Regressor** to predict prices. Key features include:
- Latitude/Longitude
- Room Type (Encoded)
- Reviews info
- Availability
