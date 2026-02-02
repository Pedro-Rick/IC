import numpy as np
import pandas as pd
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "Nabla"

BASE_DIR = Path(__file__).resolve().parent
PATH = BASE_DIR / ".." / ".." / "dataset" / MOTOR

TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE  = "_all_scaled_test.csv"

train_data = pd.DataFrame()
train_data = pd.concat([train_data,pd.read_csv(PATH / f"idiq{TRAIN_FILE}").drop(columns="Unnamed: 0")], axis=1)
train_data["speed"] = pd.read_csv(PATH / f"speed{TRAIN_FILE}")["N"]
train_data = pd.concat([train_data,pd.read_csv(PATH / f"xgeom{TRAIN_FILE}").drop(columns="Unnamed: 0")], axis=1)
train_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TRAIN_FILE}")["total"]
train_data["joule"] = pd.read_csv(PATH / f"joule{TRAIN_FILE}")["total"]

test_data = pd.DataFrame()
test_data = pd.concat([test_data,pd.read_csv(PATH / f"idiq{TEST_FILE}").drop(columns="Unnamed: 0")], axis=1)
test_data["speed"] = pd.read_csv(PATH / f"speed{TEST_FILE}")["N"]
test_data = pd.concat([test_data,pd.read_csv(PATH / f"xgeom{TEST_FILE}").drop(columns="Unnamed: 0")], axis=1)
test_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TEST_FILE}")["total"]
test_data["joule"] = pd.read_csv(PATH / f"joule{TEST_FILE}")["total"]

class TansLRegressionModel(nn.Module):

    def __init__(self, output_dim, ft_neurons, ft_layers, peso_path):
        super().__init__()

        # carregar modelo completo salvo
        full_model = torch.load(peso_path, map_location="cpu")

        # remover última camada
        self.pretrained_block = nn.Sequential(
            *list(full_model.children())[:-1]
        )

        # congelar pré
        for p in self.pretrained_block.parameters():
            p.requires_grad = False

        # descobrir saída do pré automaticamente
        last_linear = [m for m in self.pretrained_block if isinstance(m, nn.Linear)][-1]
        pre_out_dim = last_linear.out_features

        # fine tuning
        ft_modules = []
        ft_modules.append(nn.Linear(pre_out_dim, ft_neurons))
        ft_modules.append(nn.ReLU())

        for _ in range(ft_layers):
            ft_modules.append(nn.Linear(ft_neurons, ft_neurons))
            ft_modules.append(nn.ReLU())

        ft_modules.append(nn.Linear(ft_neurons, output_dim))

        self.finetune_block = nn.Sequential(*ft_modules)

    def forward(self, x):
        x = self.pretrained_block(x)
        x = self.finetune_block(x)
        return x


class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def register_csv(contents, info):
    new_row = pd.DataFrame([contents], columns=info.columns)
    info = pd.concat([info, new_row])

    SAVE_PATH = BASE_DIR / ".." / "transL_results"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_PATH / "motor_Nabla_Hys_TransL_info.csv"

    info.to_csv(SAVE_PATH, index=False)
    return info

target = ["hysteresis"]

BATCH_SIZE = 256

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# buscando o arquivo
arquivo = Path(BASE_DIR / ".." / "data_pesos" / "pesos_V_Hys")

#para finetuning

ft_neurons = np.arange(10, 200 + 1, 10)
ft_layers = [1, 2]
ft_learning_rates = [0.1, 0.01]
epochs = 100

columns = [ "ft_neurons", "ft_layers", "lr", "epochs", "hys_score", "hys_mse", "hys_mape", "time"]

info = pd.DataFrame(columns=columns)

for i in range(len(ft_neurons)):
    for j in range(len(ft_layers)):
        for k in range(len(ft_learning_rates)):

            print(f"\nTraining model --- {ft_neurons[i]}-{ft_layers[j]}-{ft_learning_rates[k]}-{epochs}\n")

            model = TansLRegressionModel(
                output_dim=1,
                ft_neurons=ft_neurons[i],
                ft_layers=ft_layers[j],
                peso_path= arquivo 
            )

            loss_func = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = ft_learning_rates[k])
             
            for a in range(epochs):
                model.train()
                for X, y in train_loader:
                    pred_train = model(X)
                    loss = loss_func(pred_train, y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            time = datetime.datetime.now()

            y_pred_list = []
            y_test_list = []

            model.eval()

            with torch.no_grad():
                for X, y in test_loader:
                    pred_test = model(X)
                    y_pred_list.append(pred_test)
                    y_test_list.append(y)

            y_pred = torch.cat(y_pred_list)
            y_test = torch.cat(y_test_list)

            hys_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
            hys_mse = mean_squared_error(y_test.detach().numpy(), y_pred.detach().numpy())
            hys_mape = mean_absolute_percentage_error(y_test.detach().numpy(), y_pred.detach().numpy())


            print(f"R2={hys_score:.4f} | MSE={hys_mse:.4e} | MAPE={hys_mape:.4f}")

            contents = [ft_neurons[i], ft_layers[j], ft_learning_rates[k], epochs, hys_score, hys_mse, hys_mape, time]

            info = register_csv(contents, info)

print("\nFIM")