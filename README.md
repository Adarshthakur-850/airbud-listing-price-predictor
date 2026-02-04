# Airbnb Listing Price Predictor

A machine learning project for predicting Airbnb listing prices using historical listing data and regression modeling. This repository contains code, data processing, model training, evaluation, and utilities to build a predictive pricing model.

## 🚀 Project Overview  

Price optimization is a key challenge for Airbnb hosts — setting a price too high may reduce bookings, while a price too low may reduce revenue. This project uses machine learning to estimate optimal listing prices based on property attributes such as location, number of rooms, amenities, and review statistics.

The goal of this repository is to provide a reproducible workflow for:
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and evaluation
- Price prediction

## 📁 Repository Structure 

```

├── data/
│   ├── raw/                   # Raw datasets used for training
│   ├── processed/             # Cleaned and feature-engineered datasets
├── notebooks/
│   └── eda_and_modeling.ipynb # EDA and model development notebooks
├── models/                    # Saved trained models and artifacts
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── train_model.py         # Train and save regression model
│   ├── evaluate.py            # Model evaluation scripts
│   └── predict.py             # Predict prices on new inputs
├── requirements.txt           # Project dependencies
└── README.md

````

## 🧠 Key Features

- **Data Cleaning & Feature Engineering:** Processes raw Airbnb listings into usable features.
- **Regression Modeling:** Train and evaluate regression models to estimate nightly prices.
- **Model Evaluation:** Compare performance across algorithms and visualize results.
- **Prediction Pipeline:** Make price predictions on new listings.

## 📌 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You can create a virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Data Preprocessing

```bash
python src/data_preprocessing.py
```

### Train the Model

```bash
python src/train_model.py
```

### Evaluate the Model

```bash
python src/evaluate.py
```

### Make Predictions

```bash
python src/predict.py --input_path your_input_file.csv
```

## 🧪 Evaluation Metrics

Common metrics for regression evaluation in this project include:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

These help quantify how well the model estimates listing prices compared to actual values.

## 📊 Example Workflow

1. Load and inspect the dataset in a Jupyter notebook.
2. Perform exploratory analysis (missing values, distributions, correlations).
3. Engineer new features (e.g., amenities count, neighborhood encoding).
4. Train regression models (Linear Regression, Random Forest, Gradient Boosting, etc.).
5. Evaluate and select the best-performing model.
6. Save and use the model for prediction.

## 📦 Dependencies

Dependencies are listed in `requirements.txt`. Typical libraries include:

* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `joblib` (for model serialization)

Install with:

```bash
pip install -r requirements.txt
```

## 🤝 Contributions

Contributions are welcome. If you:

* find bugs,
* want to improve model performance,
* add visualization,
* or extend this predictor to deploy as a web app,

please open an issue or submit a pull request.

