import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import prepare_data
from model import ANN

def evaluate():
    # 1. Load data & tensors
    _, _, X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors, scaler_y, pjme_data_test = prepare_data()

    # 2. Instantiate model & load best weights
    Model = ANN(input_dim=X_train_tensors.shape[1])
    Model.load_state_dict(torch.load("best_model.pt"))
    Model.eval()

    # 3. Generate predictions
    with torch.no_grad():
        test_preds = Model(X_test_tensors)

    # 4. Inverse transform predictions and actual targets back to Megawatts (MW)
    predictions_mw = scaler_y.inverse_transform(test_preds.numpy())
    y_test_mw = scaler_y.inverse_transform(y_test_tensors.numpy())

    # 5. Calculate Metrics
    mae = mean_absolute_error(y_test_mw, predictions_mw)
    rmse = np.sqrt(mean_squared_error(y_test_mw, predictions_mw))
    r2 = r2_score(y_test_mw, predictions_mw)

    print("--- Model Performance on Unseen Test Data ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} MW")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} MW")
    print(f"R² Score: {r2:.4f}")

    # 6. Build Results DataFrame
    results_df = pd.DataFrame({
        'Actual Load (MW)': y_test_mw.flatten(),
        'Predicted Load (MW)': predictions_mw.flatten()
    }, index=pjme_data_test.index)
    results_df['Difference (MW)'] = (results_df['Actual Load (MW)'] - results_df['Predicted Load (MW)']).abs()

    # 7. Save Actual vs Predicted Forecast Plot for GitHub README
    plt.figure(figsize=(15, 6))
    plt.plot(results_df.index[:200], results_df['Actual Load (MW)'][:200], label='Actual Load', color='#1f77b4', linewidth=2)
    plt.plot(results_df.index[:200], results_df['Predicted Load (MW)'][:200], label='ANN Forecast', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title('PJM Energy Load Forecasting - Actual vs. Neural Network Predictions (Test Set)', fontsize=14, pad=15)
    plt.xlabel('Datetime', fontsize=12)
    plt.ylabel('Megawatts (MW)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    evaluate()