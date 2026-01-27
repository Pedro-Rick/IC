import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path


# ======================
# DATA LOADING
# ======================

MOTOR = "Nabla"

BASE_DIR = Path(__file__).resolve().parent
PATH = BASE_DIR / ".." / ".." / "dataset" / MOTOR

TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE  = "_all_scaled_test.csv"

train_data = pd.DataFrame()
train_data = pd.concat([train_data, pd.read_csv(PATH / f"idiq{TRAIN_FILE}").drop(columns="Unnamed: 0")], axis=1)
train_data["speed"] = pd.read_csv(PATH / f"speed{TRAIN_FILE}")["N"]
train_data = pd.concat([train_data, pd.read_csv(PATH / f"xgeom{TRAIN_FILE}").drop(columns="Unnamed: 0")], axis=1)
train_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TRAIN_FILE}")["total"]
train_data["joule"] = pd.read_csv(PATH / f"joule{TRAIN_FILE}")["total"]

test_data = pd.DataFrame()
test_data = pd.concat([test_data, pd.read_csv(PATH / f"idiq{TEST_FILE}").drop(columns="Unnamed: 0")], axis=1)
test_data["speed"] = pd.read_csv(PATH / f"speed{TEST_FILE}")["N"]
test_data = pd.concat([test_data, pd.read_csv(PATH / f"xgeom{TEST_FILE}").drop(columns="Unnamed: 0")], axis=1)
test_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TEST_FILE}")["total"]
test_data["joule"] = pd.read_csv(PATH / f"joule{TEST_FILE}")["total"]


# ======================
# MODELS
# ======================

class transL(nn.Module):
    def __init__(self, input_dim, neurons, layers):
        super().__init__()

        modules = [nn.Linear(input_dim, neurons), nn.ReLU()]
        for _ in range(layers):
            modules += [nn.Linear(neurons, neurons), nn.ReLU()]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class mixedModel(nn.Module):
    def __init__(self, input_dim_model2, output_dim,
                 transL_input_dim, transL_neurons, transL_layers,
                 head_neurons, head_layers):
        super().__init__()

        self.input_adapter = nn.Sequential(
            nn.Linear(input_dim_model2, transL_input_dim),
            nn.ReLU()
        )

        self.transL = transL(
            transL_input_dim, transL_neurons, transL_layers
        )

        head_modules = []
        in_dim = transL_neurons
        for _ in range(head_layers):
            head_modules += [nn.Linear(in_dim, head_neurons), nn.ReLU()]
            in_dim = head_neurons

        head_modules.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*head_modules)

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.transL(x)
        return self.head(x)


# ======================
# DATASET
# ======================

class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======================
# TRAIN SETUP
# ======================

target = ['hysteresis']

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

input_dim = train_data.drop(columns=target).shape[1]


# ======================
# TRANSFER SETTINGS (FIXOS)
# ======================

TRANS_NEURONS = 140
TRANS_LAYERS  = 1
TRANS_WEIGHTS = "pesos_V_Hys_neurons140_layers1.pt"


# ======================
# GRID (SÓ DO HEAD)
# ======================

head_neurons = [16, 32, 64]
head_layers  = [1, 2]
learning_rates = [1e-3, 5e-4]
epochs = 100

columns = ['head_neurons', 'head_layers', 'lr', 'epochs',
           'hys_score', 'hys_mse', 'hys_mape', 'time']

info = pd.DataFrame(columns=columns)


# ======================
# TRAIN LOOP
# ======================

for hn in head_neurons:
    for hl in head_layers:
        for lr in learning_rates:

            print(f"\nTraining TL model — head {hn}x{hl}, lr={lr}\n")

            model = mixedModel(
                input_dim_model2=input_dim,
                output_dim=1,
                transL_input_dim=input_dim,
                transL_neurons=TRANS_NEURONS,
                transL_layers=TRANS_LAYERS,
                head_neurons=hn,
                head_layers=hl
            )

            model.transL.load_state_dict(
                torch.load(TRANS_WEIGHTS, map_location="cpu"),
                strict=False
            )

            for param in model.transL.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr
            )

            loss_func = nn.MSELoss()

            for _ in range(epochs):
                model.train()
                for X, y in train_loader:
                    optimizer.zero_grad()
                    loss = loss_func(model(X), y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            y_pred, y_true = [], []

            with torch.no_grad():
                for X, y in test_loader:
                    y_pred.append(model(X))
                    y_true.append(y)

            y_pred = torch.cat(y_pred)
            y_true = torch.cat(y_true)

            hys_score = r2_score(y_true.numpy(), y_pred.numpy())
            hys_mse   = mean_squared_error(y_true.numpy(), y_pred.numpy())
            hys_mape  = mean_absolute_percentage_error(y_true.numpy(), y_pred.numpy())

            time = datetime.datetime.now()

            info.loc[len(info)] = [
                hn, hl, lr, epochs,
                hys_score, hys_mse, hys_mape, time
            ]

            print(f"R2={hys_score:.4f} | MSE={hys_mse:.4e} | MAPE={hys_mape:.4f}")


print("\n=== FIM ===")
