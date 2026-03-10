import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "Nabla"
MOTOR_TL = "2D"
var = "Hys"
target = ["hysteresis"]
curve_parameters = ["MAPE", "RMSE"]

BASE_DIR = Path(__file__).resolve().parent
IC_BASE_DIR = BASE_DIR.parent.parent
PATH = IC_BASE_DIR / "dataset" / MOTOR

TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE  = "_all_scaled_test.csv"

# =========================
# LOAD DATA
# =========================

train_data = pd.DataFrame()

train_data = pd.concat([train_data, pd.read_csv(PATH / f"idiq{TRAIN_FILE}").drop(columns="Unnamed: 0")], axis=1)
train_data["speed"] = pd.read_csv(PATH / f"speed{TRAIN_FILE}")["N"]
train_data = pd.concat([train_data, pd.read_csv(PATH / f"xgeom{TRAIN_FILE}").drop(columns="Unnamed: 0")], axis=1)
train_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TRAIN_FILE}")["total"]
train_data["joule"]      = pd.read_csv(PATH / f"joule{TRAIN_FILE}")["total"]

test_data = pd.DataFrame()

test_data = pd.concat([test_data, pd.read_csv(PATH / f"idiq{TEST_FILE}").drop(columns="Unnamed: 0")], axis=1)
test_data["speed"] = pd.read_csv(PATH / f"speed{TEST_FILE}")["N"]
test_data = pd.concat([test_data, pd.read_csv(PATH / f"xgeom{TEST_FILE}").drop(columns="Unnamed: 0")], axis=1)
test_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TEST_FILE}")["total"]
test_data["joule"]      = pd.read_csv(PATH / f"joule{TEST_FILE}")["total"]

# =========================
# TL MODEL
# =========================

class TLRegressionModel(nn.Module):

    def __init__(self, input_dim, peso_path, unlock_layers):
        super().__init__()

        full_model = torch.load(peso_path, map_location="cpu", weights_only=False)
        self.pretrained_block = full_model

        pre_input_dim = self.pretrained_block[0].in_features

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, pre_input_dim),
            nn.ReLU()
        )

        for p in self.pretrained_block.parameters():
            p.requires_grad = False

        layers = [layer for layer in self.pretrained_block if len(list(layer.parameters())) > 0]

        if unlock_layers > 0:
            for layer in layers[-unlock_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, x):

        x = self.adapter(x)
        x = self.pretrained_block(x)

        return x


# =========================
# BASE MODEL
# =========================

class RegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim, neurons=5, layers=1):
        super().__init__()

        modules = []

        modules.append(nn.Linear(input_dim, neurons))
        modules.append(nn.ReLU())

        for _ in range(layers):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(neurons, output_dim))

        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        return self.linear(x)


# =========================
# DATASET
# =========================

class MotorDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# =========================
# REGISTER
# =========================

def register_csv(contents, info, arq_name):

    new_row = pd.DataFrame([contents], columns=info.columns)
    info = pd.concat([info, new_row], ignore_index=True)

    SAVE_PATH = IC_BASE_DIR / arq_name

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    info.to_csv(SAVE_PATH, index=False)

    return info


# =========================
# BEST PARAMS
# =========================

def get_best_mape_row(parameter):

    csv_path = IC_BASE_DIR / "results_patu" / f"{MOTOR_TL}" / f"motor_{MOTOR_TL}_{var}_info.csv"
    df = pd.read_csv(csv_path)

    idx = df[f"{var}_mape"].idxmin()

    best_row = df.loc[idx]

    return best_row[parameter]


# =========================
# DATA LOADERS
# =========================

BATCH_SIZE = 256

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

train_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = len(train_data.columns.drop(target))

ARQ_PESOS = IC_BASE_DIR / "transferLearning" / "data_pesos" / f"pesos_{MOTOR_TL}_{var}.pt"

# =========================
# COUNT TRAINABLE LAYERS
# =========================
temp_model = torch.load(ARQ_PESOS, map_location="cpu", weights_only=False)

trainable_layers = []

for name, param in temp_model.named_parameters():

    # pega só o nome da camada (ex: "0.weight" → "0")
    layer_name = name.split(".")[0]

    if layer_name not in trainable_layers:
        trainable_layers.append(layer_name)

print("\nCamadas treináveis encontradas:", trainable_layers)

max_unlock = len(trainable_layers)

print("Máximo unlock_layers possível:", max_unlock)

# =========================
# CONFIG
# =========================
unlock_layers_list = list(range(0, max_unlock + 1))

epochs = 100
models = ["TLap", "TLa"]

b_TL_lr = float(get_best_mape_row("learn_rate"))
b_TL_neurons = int(get_best_mape_row("neurons"))
b_TL_layers = int(get_best_mape_row("layers"))

neurons_max= int(b_TL_neurons * b_TL_layers)

plt.figure()

# ========================
# MAIN LOOP
# =========================

results_curves = {}

for model_type in models:

    print("==========")
    print(f"modelo: {model_type}")
    print("==========")
  
    # =========================
    # TL WITH WEIGHTS
    # =========================

    if model_type == "TLap":

        best_epoch_mape = float("inf")
        best_epoch_rmse = float("inf")

        best_curve_mape = None
        best_curve_rmse = None
        best_unlock = None

        SAVE_CONTS = Path("transferLearning") / "TL_results"/ f"{MOTOR}" / f"motor_{MOTOR}_{var}_TL_{MOTOR_TL}_arq_wei_info.csv"

        columns = ["neurons","layers","lr","unlock_layers","epochs",
                   f"{var}_score",f"{var}_mse",f"{var}_rmse",f"{var}_mape","time"]

        info = pd.DataFrame(columns=columns)

        for unlock in unlock_layers_list:

            print(f"\nTestando unlock_layers = {unlock}")

            model = TLRegressionModel(
                input_dim=input_dim,
                peso_path=ARQ_PESOS,
                unlock_layers=unlock
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            loss_func = nn.MSELoss()

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=b_TL_lr
            )

            mape_curve = []
            rmse_curve = []

            start_time = datetime.datetime.now()

            for ep in range(epochs):

                model.train()

                for X,y in train_loader_full:

                    X,y = X.to(device),y.to(device)

                    pred = model(X)
                    loss = loss_func(pred,y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                y_pred_list=[]
                y_test_list=[]

                model.eval()

                with torch.no_grad():

                    for X,y in test_loader:

                        X,y = X.to(device),y.to(device)

                        y_pred_list.append(model(X))
                        y_test_list.append(y)

                y_pred=torch.cat(y_pred_list).cpu()
                y_test=torch.cat(y_test_list).cpu()

                Hys_mse = mean_squared_error(y_test.numpy(),y_pred.numpy())
                Hys_rmse = np.sqrt(Hys_mse)
                Hys_mape = mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

                mape_curve.append(Hys_mape)
                rmse_curve.append(Hys_rmse)

            # comparação usando epoch final
            if mape_curve[-1] < best_epoch_mape:
                best_epoch_mape = mape_curve[-1]
                best_curve_mape = mape_curve.copy()
                best_unlock = unlock

            if rmse_curve[-1] < best_epoch_rmse:
                best_epoch_rmse = rmse_curve[-1]
                best_curve_rmse = rmse_curve.copy()

            Hys_score=r2_score(y_test.numpy(),y_pred.numpy())

            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            contents = [b_TL_neurons,b_TL_layers,b_TL_lr,unlock,epochs,
                        Hys_score,Hys_mse,Hys_rmse,Hys_mape,elapsed_time]

            info = register_csv(contents, info, SAVE_CONTS)

        print(f"\nMelhor unlock_layers = {best_unlock}")

        results_curves["TLap_MAPE"] = best_curve_mape
        results_curves["TLap_RMSE"] = best_curve_rmse

    # =========================
    # TL ARCHITECTURE
    # =========================

    if model_type == "TLa":

        model = RegressionModel(
            input_dim=input_dim,
            output_dim=1,
            neurons=b_TL_neurons,
            layers=b_TL_layers
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=b_TL_lr)

        mape_curve = []
        rmse_curve = []

        start_time = datetime.datetime.now()

        for ep in range(epochs):

            model.train()

            for X,y in train_loader_full:

                X,y = X.to(device),y.to(device)

                pred=model(X)
                loss=loss_func(pred,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            y_pred_list=[]
            y_test_list=[]

            model.eval()

            with torch.no_grad():

                for X,y in test_loader:

                    X,y = X.to(device),y.to(device)

                    y_pred_list.append(model(X))
                    y_test_list.append(y)

            y_pred=torch.cat(y_pred_list).cpu()
            y_test=torch.cat(y_test_list).cpu()

            Hys_mse=mean_squared_error(y_test.numpy(),y_pred.numpy())
            Hys_rmse=np.sqrt(Hys_mse)
            Hys_mape=mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

            mape_curve.append(Hys_mape)
            rmse_curve.append(Hys_rmse)

        results_curves["TLa_MAPE"] = mape_curve
        results_curves["TLa_RMSE"] = rmse_curve

        SAVE_CONTS = Path("transferLearning") / "TL_results" / f"{MOTOR}" / f"motor_{MOTOR}_{var}_TL_{MOTOR_TL}_arq_info.csv"

        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        columns = ["neurons","layers","lr","epochs",
                   f"{var}_score",f"{var}_mse",f"{var}_rmse",f"{var}_mape","time"]

        info = pd.DataFrame(columns=columns)

        contents = [b_TL_neurons,b_TL_layers,b_TL_lr,epochs,
                    Hys_score,Hys_mse,Hys_rmse,Hys_mape,elapsed_time]

        info = register_csv(contents, info, SAVE_CONTS)


# =========================
# SAVE + PLOT CURVES
# =========================

for curve_var in curve_parameters:

    plt.figure()

    for name,curve in results_curves.items():

        if curve_var not in name:
            continue

        curve_df=pd.DataFrame({
            "epoch":np.arange(1,epochs+1),
            curve_var.lower():curve
        })

        curve_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / "graficos" / f"curve_{name}_{MOTOR}_{var}.csv"

        curve_path.parent.mkdir(parents=True,exist_ok=True)

        curve_df.to_csv(curve_path,index=False)

        print("Curva TL salva em:",curve_path)

        plt.plot(
            curve_df["epoch"],
            curve_df[curve_var.lower()],
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel(curve_var)

    plt.title(f"{MOTOR}_{var} — TLa vs TLap - {curve_var}")

    plt.grid(True)
    plt.legend()

    save_fig = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / "graficos"

    save_fig.mkdir(parents=True,exist_ok=True)

    plt.savefig(save_fig / f"TLa_vs_TLap-{MOTOR}_TL_{MOTOR_TL}_{var}_{curve_var}.png")

    plt.show()

print("\nFIM")