import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()

        self.model = nn.Sequential(

            # 1st hidden layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # 2nd hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # output layer
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)