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
# TL_ARQ_WEIGHTS_MODEL
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

        for layer in list(self.pretrained_block.children())[-(unlock_layers):]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x):

        x = self.adapter(x)
        x = self.pretrained_block(x)
        return x
    
# =========================
# TL_ARQ_MODEL
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

    # cria só a pasta
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    info.to_csv(SAVE_PATH, index=False)

    return info

# =========================
# BEST_ARQ_TL
# =========================
def get_best_mape_row(parameter):
    
    csv_path = IC_BASE_DIR / "results_patu" / f"{MOTOR_TL}" / f"motor_{MOTOR_TL}_{var}_info.csv"
    df = pd.read_csv(csv_path)

    idx = df[f"{var}_mape"].idxmin()
    best_row = df.loc[idx]

    b_parameter = best_row[parameter]
    
    return b_parameter

# =========================
# DATA LOADERS
# =========================
BATCH_SIZE = 256

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

train_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

ARQ_PESOS = IC_BASE_DIR / "transferLearning" / "data_pesos" / f"pesos_{MOTOR_TL}_{var}.pt"

# =========================
# CONFIG
# =========================
unlock_layers = []
epochs = 100
models = ["TLap", "TLa"]
b_TL_lr = float(get_best_mape_row("learn_rate"))
b_TL_neurons = int(get_best_mape_row("neurons"))
b_TL_layers = int(get_best_mape_row("layers"))

columns = ["neurons", "layers", "lr","epochs", f"{var}_score", f"{var}_mse", f"{var}_mape", "time"]
info = pd.DataFrame(columns=columns)

# =========================
# CURVA GLOBAL
# =========================
epoch_curve = np.zeros(epochs)
best_global_mape = float("inf")
best_model_block = None

# =========================
# MAIN LOOP
# =========================

best_global_mape = float("inf")
best_curve_global = None

all_curves = []
plt.figure()


# ===== TRAIN POR EPOCH =====
for model_type in models:

    if model_type == "TLa":
        model = RegressionModel(input_dim=len(train_data.columns.drop(target)), output_dim=1, neurons=b_TL_neurons, layers=b_TL_layers)
        
        SAVE_CONTS = Path("transferLearning") / "TL_results" / f"{MOTOR}" / f"motor_{MOTOR}_{var}_TL_{MOTOR_TL}_arq_info.csv"

        curve_name_csv = f"curve_TL_epochs_{MOTOR}_{var}_arq.csv"
        curve_name = "TL_arq"

    if model_type == "TLap":
        
        model = TLRegressionModel(input_dim=len(train_data.columns.drop(target)), peso_path=ARQ_PESOS, unlock_layers=1)

        SAVE_CONTS = Path("transferLearning") / "TL_results"/ f"{MOTOR}" / f"motor_{MOTOR}_{var}_TL_{MOTOR_TL}_arq_wei_info.csv"

        curve_name_csv = f"curve_TL_epochs_{MOTOR}_{var}_arq_wei.csv"
        curve_name = "TL_arq_weitghs"

    print("==========")
    print(f"modelo: {model_type}")
    print("==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=b_TL_lr
    )

    start_time = datetime.datetime.now()

    best_mape = float("inf")
    best_mape_so_far = []

    for ep in range(epochs):

        print(f"======= Epoca {ep+1} =======")

        model.train()
        for X, y in train_loader_full:
            X, y = X.to(device), y.to(device)

            pred_train = model(X)
            loss = loss_func(pred_train, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # ===== EVAL =====
        y_pred_list = []
        y_test_list = []

        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred_list.append(model(X))
                y_test_list.append(y)

        y_pred = torch.cat(y_pred_list).cpu()
        y_test = torch.cat(y_test_list).cpu()

        Jou_score = r2_score(y_test.numpy(), y_pred.numpy())
        Jou_mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
        Jou_mape = mean_absolute_percentage_error(y_test.numpy(), y_pred.numpy())

        print(f"")

        if Jou_mape < best_mape:
            best_mape = Jou_mape

        print(f"best_mape = {best_mape} || Jou_mape = {Jou_mape}")
        print("")

        best_mape_so_far.append(Jou_mape)

    if best_mape < best_global_mape:
        best_global_mape = best_mape
        best_curve_global = best_mape_so_far.copy()

        if model_type == "TLap":
            best_model_block = model.pretrained_block

    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    contents = [b_TL_neurons, b_TL_layers, b_TL_lr, epochs, Jou_score, Jou_mse, Jou_mape, elapsed_time]
    info = register_csv(contents, info, SAVE_CONTS)

    # =========================
    # SAVE CURVE
    # =========================
    curve_df = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "mape": best_curve_global,
    })

    curve_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / "graficos" / curve_name_csv

    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(curve_path, index=False)
    print("Curva TL salva em:", curve_path)

    # =========================
    # PLOT 
    # =========================
    plt.plot(
    curve_df["epoch"],
    curve_df["mape"],
    label= curve_name,
    )

# =========================
# LOAD BASELINE (EPOCHS)
# =========================
baseline_path = IC_BASE_DIR / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_epochs_{MOTOR}_{var}.csv"
base_df = pd.read_csv(baseline_path)

# =========================
# PLOT BASELINE
# =========================

plt.plot(
    base_df["epoch"],
    base_df["mape"],
    label="Baseline",
)

plt.xlabel("Epoch")
plt.ylabel("MAPE")
plt.title(f"{MOTOR}_{var} — Baseline vs TLa vs TLap")
plt.grid(True)
plt.legend()

save_fig =IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / "graficos"
save_fig.mkdir(parents=True, exist_ok=True)

plt.savefig(save_fig / f"baseline_TLa_TLap_{MOTOR}_TL_{MOTOR_TL}_{var}.png")
plt.show()

print("\nFIM")