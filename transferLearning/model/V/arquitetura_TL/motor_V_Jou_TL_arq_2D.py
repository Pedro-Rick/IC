import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "V"
MOTOR_TL = "2D"
var = "Jou"
target = ["joule"]
curve_parameters = ["MAPE", "RMSE"]

BASE_DIR = Path(__file__).resolve().parent
IC_BASE_DIR = BASE_DIR.parent.parent.parent.parent
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
# MODEL
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
# PARAMETERS
# =========================

b_TL_lr = float(get_best_mape_row("learn_rate"))
b_TL_neurons = int(get_best_mape_row("neurons"))
b_TL_layers = int(get_best_mape_row("layers"))

neurons_max= int(b_TL_neurons * b_TL_layers)

results_curves = {}

columns = ["neurons","layers","lr","epochs", f"{var}_score",f"{var}_mse",f"{var}_rmse",f"{var}_mape","time"]
info = pd.DataFrame(columns=columns)

media_mape_epochs = []
media_rmse_epochs = []

# =========================
# CONFIG
# =========================
epochs = 100
seeds = 30

# =========================
# TL ARCHITECTURE
# =========================
for seed in range (seeds):

    print("")
    print(f"===== Seed {seed + 1} =====")
    print("")
    
    model = RegressionModel(input_dim=input_dim, output_dim=1, neurons=b_TL_neurons, layers=b_TL_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=b_TL_lr)

    mape_epochs = []
    rmse_epochs = []

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
        
        Jou_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
        Jou_mse=mean_squared_error(y_test.numpy(),y_pred.numpy())
        Jou_rmse=np.sqrt(Jou_mse)
        Jou_mape=mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

        mape_epochs.append(Jou_mape)
        rmse_epochs.append(Jou_rmse)

        if ((((ep+1)%50)== 0) or ((ep) == 0)):
            print(f"Epoch {ep +1} || MAPE = {Jou_mape}  RMSE = {Jou_rmse}")
    
    media_mape_epochs.append(mape_epochs)
    media_rmse_epochs.append(rmse_epochs)

mape_array = np.array(media_mape_epochs)
rmse_array = np.array(media_rmse_epochs)

mape_mean = np.mean(mape_array, axis=0)
mape_std  = np.std(mape_array, axis=0)

rmse_mean = np.mean(rmse_array, axis=0)
rmse_std  = np.std(rmse_array, axis=0)

results_curves["TLa_MAPE"] = (mape_mean, mape_std)
results_curves["TLa_RMSE"] = (rmse_mean, rmse_std)

SAVE_CONTS = Path("transferLearning") / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / f"{MOTOR}_TL_arq_{MOTOR_TL}_{var}_info.csv"

end_time = datetime.datetime.now()
elapsed_time = (end_time - start_time).total_seconds()

contents = [b_TL_neurons,b_TL_layers,b_TL_lr,epochs, Jou_score,Jou_mse,Jou_rmse,Jou_mape,elapsed_time]

info = register_csv(contents, info, SAVE_CONTS)

# =========================
# SAVE + PLOT CURVES
# =========================

for curve_var in curve_parameters:

    plt.figure()

    # ========================= 
    # LOAD BASELINE 
    # =========================
    baseline_path = IC_BASE_DIR / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_epochs_{MOTOR}_{var}_{curve_var}.csv"

    if baseline_path.exists():
        base_df = pd.read_csv(baseline_path)

        line, = plt.plot(
            base_df["epoch"],
            base_df[curve_var.lower()],
            label="Baseline"
        )

    else:
        print(f"CSV curva baseline {curve_var} ainda não existe")

    # ========================= 
    # LOAD TL PESOS 
    # =========================
    TL_PESOS_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / "graficos" / f"curve_TLap_{curve_var}_{MOTOR}_{var}.csv"

    if TL_PESOS_path.exists():
        pesos_df = pd.read_csv(TL_PESOS_path)

        line, = plt.plot(
            pesos_df["epoch"],
            pesos_df[f"{curve_var.lower()}_mean"],
            label=f"TLap_{curve_var}"
        )

        plt.fill_between(
            pesos_df["epoch"],
            pesos_df[f"{curve_var.lower()}_mean"] - pesos_df[f"{curve_var.lower()}_std"],
            pesos_df[f"{curve_var.lower()}_mean"] + pesos_df[f"{curve_var.lower()}_std"],
            color=line.get_color(),
            alpha=0.25
        )

    else:
        print(f"CSV curva pesos {curve_var} ainda não existe")

    for name,(curve_mean,curve_std) in results_curves.items():

        if curve_var not in name:
            continue

        curve_df = pd.DataFrame({
            "epoch": np.arange(1, epochs+1),
            f"{curve_var.lower()}_mean": curve_mean,
            f"{curve_var.lower()}_std": curve_std
        })

        curve_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / "graficos" / f"curve_{name}_{MOTOR}_{var}.csv"

        curve_path.parent.mkdir(parents=True,exist_ok=True)

        curve_df.to_csv(curve_path,index=False)

        print("Curva TL salva em:",curve_path)

        line, = plt.plot(
            curve_df["epoch"],
            curve_df[f"{curve_var.lower()}_mean"],
            label=name
        )

        plt.fill_between(
            curve_df["epoch"],
            curve_df[f"{curve_var.lower()}_mean"] - curve_df[f"{curve_var.lower()}_std"],
            curve_df[f"{curve_var.lower()}_mean"] + curve_df[f"{curve_var.lower()}_std"],
            color=line.get_color(),
            alpha=0.25
        )

    plt.xlabel("Epoch")
    plt.ylabel(curve_var)

    plt.title(f"{MOTOR}_{var} - TL:{MOTOR_TL} - TLa, TLap, Baseline - {curve_var} ")

    plt.grid(True)
    plt.legend()

    save_fig = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / "graficos"

    save_fig.mkdir(parents=True,exist_ok=True)

    plt.savefig(save_fig / f"baseline_TLa_TLp-{MOTOR}_TL_{MOTOR_TL}_{var}_{curve_var}.png")

    plt.show()

print("\nFIM")