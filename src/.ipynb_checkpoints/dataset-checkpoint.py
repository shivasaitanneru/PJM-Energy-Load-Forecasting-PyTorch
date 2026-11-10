import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def prepare_data(csv_path="../data/PJME_hourly.csv", target_date="2015-01-01", batch_size=48):
    # 1. Load data and set Datetime index
    pjme_data = pd.read_csv(csv_path)
    pjme_data["Datetime"] = pd.to_datetime(pjme_data["Datetime"])
    pjme_data = pjme_data.set_index("Datetime").sort_index()

    # 2. Time components
    pjme_data["hour"] = pjme_data.index.hour
    pjme_data["day_of_week"] = pjme_data.index.dayofweek
    pjme_data["quarter"] = pjme_data.index.quarter
    pjme_data["month"] = pjme_data.index.month
    pjme_data["year"] = pjme_data.index.year
    pjme_data["day_of_year"] = pjme_data.index.dayofyear

    # 3. Lag features
    pjme_data["24_lag"] = pjme_data["PJME_MW"].shift(24)
    pjme_data["1week_lag"] = pjme_data["PJME_MW"].shift(168)
    pjme_data = pjme_data.dropna()

    # 4. Cyclic features
    pjme_data["hour_sin"] = np.sin((2 * np.pi * pjme_data["hour"]) / 24)
    pjme_data["hour_cos"] = np.cos((2 * np.pi * pjme_data["hour"]) / 24)
    pjme_data["month_sin"] = np.sin((2 * np.pi * pjme_data["hour"]) / 12)
    pjme_data["month_cos"] = np.cos((2 * np.pi * pjme_data["hour"]) / 12)

    # 5. Train / Test split
    pjme_data_train = pjme_data[pjme_data.index < target_date]
    pjme_data_test = pjme_data[pjme_data.index >= target_date]

    X_train = pjme_data_train.drop(columns=["PJME_MW"])
    y_train = pjme_data_train["PJME_MW"]

    X_test = pjme_data_test.drop(columns=["PJME_MW"])
    y_test = pjme_data_test["PJME_MW"]

    # 6. Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaler = scaler_X.fit_transform(X_train)
    X_test_scaler = scaler_X.transform(X_test)

    y_train_scaler = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaler = scaler_y.transform(y_test.values.reshape(-1, 1))

    # 7. PyTorch Tensors & DataLoaders
    X_train_tensors = torch.tensor(X_train_scaler, dtype=torch.float32)
    y_train_tensors = torch.tensor(y_train_scaler, dtype=torch.float32)

    X_test_tensors = torch.tensor(X_test_scaler, dtype=torch.float32)
    y_test_tensors = torch.tensor(y_test_scaler, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
    test_dataset = TensorDataset(X_test_tensors, y_test_tensors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors, scaler_y, pjme_data_test