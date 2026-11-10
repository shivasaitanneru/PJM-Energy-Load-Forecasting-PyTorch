import torch
import torch.nn as nn
import torch.optim as optim
from dataset import prepare_data
from model import ANN

def train():
    # 1. Load Data
    train_loader, test_loader, X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors, scaler_y, pjme_data_test = prepare_data()

    # 2. Initialize Model, Criterion, and Optimizer
    Model = ANN(input_dim=X_train_tensors.shape[1])
    criterian = nn.MSELoss()
    optimizer = optim.Adam(Model.parameters(), lr=0.001)

    # 3. Training Loop
    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    epochs = 50

    for epoch in range(epochs):
        Model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = Model(xb)
            loss = criterian(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        train_loss.append(epoch_train_loss)

        # Validation
        Model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for xb, yb in test_loader:
                outputs = Model(xb)
                loss = criterian(outputs, yb)
                running_loss += loss.item()

        epoch_val_loss = running_loss / len(test_loader)
        val_loss.append(epoch_val_loss)

        print(f"epoch : {epoch+1}, epochtrainloss : {epoch_train_loss}, epochtestloss :{epoch_val_loss} ")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(Model.state_dict(), "best_model.pt")

if __name__ == "__main__":
    train()