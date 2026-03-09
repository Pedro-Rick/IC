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
var = "Jou"
target = ["joule"]

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

plt.figure()

# =========================
# MAIN LOOP
# =========================

for model_type in models:

    print("==========")
    print(f"modelo: {model_type}")
    print("==========")

    # =========================
    # TL WITH WEIGHTS
    # =========================

    if model_type == "TLap":

        best_global_mape = float("inf")
        best_curve_global = None
        best_unlock = None

        SAVE_CONTS = Path("transferLearning") / "TL_results"/ f"{MOTOR}" / f"motor_{MOTOR}_{var}_TL_{MOTOR_TL}_arq_wei_info.csv"
        columns = ["neurons", "layers", "lr", "unlock_layers", "epochs", f"{var}_score", f"{var}_mse", f"{var}_mape", "time"]
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

            best_mape = float("inf")
            mape_curve = []

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

                Jou_mape=mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

                mape_curve.append(Jou_mape)

                if Jou_mape < best_mape:
                    best_mape = Jou_mape

            if best_mape < best_global_mape:

                best_global_mape = best_mape
                best_curve_global = mape_curve.copy()
                best_unlock = unlock

            Jou_score=r2_score(y_test.numpy(),y_pred.numpy())
            Jou_mse=mean_squared_error(y_test.numpy(),y_pred.numpy())
            Jou_mape=mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            contents = [b_TL_neurons, b_TL_layers, b_TL_lr, unlock, epochs, Jou_score, Jou_mse, Jou_mape, elapsed_time]
            info = register_csv(contents, info, SAVE_CONTS)

        print(f"\nMelhor unlock_layers = {best_unlock}")

        curve_to_plot = best_curve_global

        curve_name = f"TL_arq_weights_unlock_{best_unlock}"
        curve_name_csv = f"curve_TL_epochs_{MOTOR}_{var}_arq_wei.csv"
   

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

        best_mape_so_far = []

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

            Jou_score=r2_score(y_test.numpy(),y_pred.numpy())
            Jou_mse=mean_squared_error(y_test.numpy(),y_pred.numpy())
            Jou_mape=mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

            best_mape_so_far.append(Jou_mape)
            

        curve_to_plot = best_mape_so_far

        curve_name="TL_arq"
        curve_name_csv=f"curve_TL_epochs_{MOTOR}_{var}_arq.csv"

        SAVE_CONTS = Path("transferLearning") / "TL_results" / f"{MOTOR}" / f"motor_{MOTOR}_{var}_TL_{MOTOR_TL}_arq_info.csv"

        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        columns = ["neurons", "layers", "lr","epochs", f"{var}_score", f"{var}_mse", f"{var}_mape", "time"]
        info = pd.DataFrame(columns=columns)
        contents = [b_TL_neurons, b_TL_layers, b_TL_lr, epochs, Jou_score, Jou_mse, Jou_mape, elapsed_time]
        info = register_csv(contents, info, SAVE_CONTS)

        

    # =========================
    # SAVE CURVE
    # =========================

    curve_df=pd.DataFrame({

        "epoch":np.arange(1,epochs+1),
        "mape":curve_to_plot

    })

    curve_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / "graficos" / curve_name_csv

    curve_path.parent.mkdir(parents=True,exist_ok=True)

    curve_df.to_csv(curve_path,index=False)

    print("Curva TL salva em:",curve_path)

    # =========================
    # PLOT
    # =========================

    plt.plot(
        curve_df["epoch"],
        curve_df["mape"],
        label=curve_name
    )


# =========================
# LOAD BASELINE
# =========================

#baseline_path = IC_BASE_DIR / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_epochs_{MOTOR}_{var}.csv"

#base_df = pd.read_csv(baseline_path)

# =========================
# PLOT BASELINE
# =========================

#plt.plot(
#    base_df["epoch"],
#    base_df["mape"],
#    label="Baseline"
#)

plt.xlabel("Epoch")
plt.ylabel("MAPE")

#plotar c baseline
# plt.title(f"{MOTOR}_{var} — Baseline vs TLa vs TLap") 
plt.title(f"{MOTOR}_{var} — TLa vs TLap")

plt.grid(True)
plt.legend()

save_fig = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / "graficos"

save_fig.mkdir(parents=True,exist_ok=True)

plt.savefig(save_fig / f"TLa_vs_TLap-{MOTOR}_TL_{MOTOR_TL}_{var}.png")

plt.show()

print("\nFIM")