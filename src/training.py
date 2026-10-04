import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from preprocess import load_and_preprocess_data

def evaluate_model():
    """Evaluate the performance of Ridge and Random Forest models on the wine quality dataset."""
    
    # === Load and preprocess the data ===
    data = load_and_preprocess_data()
    X_tr, y_tr = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_te, y_te = data['X_test'], data['y_test']
    
    # === Method 1: Ridge Regression ===
    alphas = np.logspace(-3, 3, 20)
    ridge_tr_errors, ridge_val_errors = [], []
    
    best_ridge_alpha = None
    best_ridge_val_mse = float('inf')
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_tr, y_tr)
        
        tr_mse = mean_squared_error(y_tr, ridge.predict(X_tr))
        val_mse = mean_squared_error(y_val, ridge.predict(X_val))
        
        ridge_tr_errors.append(tr_mse)
        ridge_val_errors.append(val_mse)
        
        if val_mse < best_ridge_val_mse:
            best_ridge_val_mse = val_mse
            best_ridge_alpha = alpha

    # === Method 2: Random Forest Regression ===
    depths = [3, 5, 8, 12, 16, 20, None]
    rf_tr_errors, rf_val_errors = [], []
    
    best_rf_depth = None
    best_rf_val_mse = float('inf')
    best_rf_model = None
    
    for depth in depths:
        rf = RandomForestRegressor(max_depth=depth, random_state=42)