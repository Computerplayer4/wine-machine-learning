import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

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
    best_ridge_model = None
    
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
            best_ridge_model = ridge

    # === Method 2: Random Forest Regression ===
    depths = [3, 5, 8, 12, 16, 20, None]
    depths_labels = [str(d) for d in depths]
    rf_tr_errors, rf_val_errors = [], []
    
    best_rf_depth = None
    best_rf_val_mse = float('inf')
    best_rf_model = None
    
    for depth in depths:
        rf = RandomForestRegressor(max_depth=depth, random_state=42)
        rf.fit(X_tr, y_tr)

        tr_mse = mean_squared_error(y_tr, rf.predict(X_tr))
        val_mse = mean_squared_error(y_val, rf.predict(X_val))
        
        rf_tr_errors.append(tr_mse)
        rf_val_errors.append(val_mse)
        
        if val_mse < best_rf_val_mse:
            best_rf_val_mse = val_mse
            best_rf_depth = depth
            best_rf_model = rf

    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Ridge Regression Validation Curve
    plt.figure(figsize=(7, 4))
    plt.semilogx(alphas, ridge_tr_errors, label='Train MSE', color='blue', linestyle='--')
    plt.semilogx(alphas, ridge_val_errors, label='Validation MSE', color='red')
    plt.axvline(best_ridge_alpha, color='black', linestyle=':', label=f'Best Alpha ({best_ridge_alpha:.2f})')
    plt.title('Ridge Regression: Validation Curve')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(assets_dir / 'ridge_validation_curve.png')
    plt.show()

    # Random Forest Regression Validation Curve
    plt.figure(figsize=(7, 4))
    x_indices = range(len(depths))
    plt.plot(x_indices, rf_tr_errors, label='Train MSE', color='blue', linestyle='--')
    plt.plot(x_indices, rf_val_errors, label='Validation MSE', color='red')
    plt.xticks(x_indices, depths_labels)
    plt.title('Random Forest Regression: Validation Curve')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(assets_dir / 'rf_validation_curve.png')
    plt.show()
    
    print("=== Model Validation Summary ===")
    print(f"Best Ridge Regression Alpha: {best_ridge_alpha:.4f} with Validation MSE: {best_ridge_val_mse:.4f}")
    print(f"Best Random Forest Max Depth: {best_rf_depth} with Validation MSE: {best_rf_val_mse:.4f}")
    
    if best_rf_val_mse < best_ridge_val_mse:
        winning_name = "Random Forest Regression"
        winning_model = best_rf_model
        winning_val_mse = best_rf_val_mse
    else:
        winning_name = "Ridge Regression"
        winning_model = best_ridge_model
        winning_val_mse = best_ridge_val_mse
    
    print(f"Selected Model for Testing: {winning_name} with Validation MSE: {winning_val_mse:.4f}")
    
    test_preds = winning_model.predict(X_te)
    test_mse = mean_squared_error(y_te, test_preds)
    test_mae = mean_absolute_error(y_te, test_preds)
    test_r2 = r2_score(y_te, test_preds)
    
    print("\n=== Test Set Performance ===")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R^2 Score: {test_r2:.4f}")
    
    save_visual_results_table(
        best_ridge_val_mse, best_rf_val_mse, winning_name, 
        test_mse, test_mae, test_r2, best_ridge_alpha, best_rf_depth
    )
   
def save_visual_results_table(ridge_val_mse, rf_val_mse, winning_name, test_mse, test_mae, test_r2, best_ridge_alpha, best_rf_depth):
    """Save the performance metrics as a table in the assets directory"""
    project_root = Path(__file__).resolve().parent.parent
    assets_dir = project_root / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)

    rf_depth_str = f"max_depth = {best_rf_depth}" if best_rf_depth is not None else "max_depth = None"
    
    data_matrix = [
        ["Ridge Regression", f"alpha = {best_ridge_alpha:.4f}", f"{ridge_val_mse:.4f}", "—", "—", "—"],
        [f"{winning_name}", rf_depth_str, f"{rf_val_mse:.4f}", f"{test_mse:.4f}", f"{test_mae:.4f}", f"{test_r2:.4f}"]
    ]
    columns = ["Model Class", "Best Hyperparameter", "Validation MSE", "Test MSE", "Test MAE", "Test R²"]
    
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data_matrix, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
        else:
            cell.set_facecolor('#E8F8F5' if row == 2 else '#F8F9F9')
    
    plt.title("Model Performance Summary", fontsize=12, weight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(assets_dir / 'model_performance_summary.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()