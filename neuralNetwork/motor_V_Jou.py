import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "V"
var = "Jou"
target = ['joule']


BASE_DIR = Path(__file__).resolve().parent
PATH = BASE_DIR.parent / "dataset" / MOTOR

TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE  = "_all_scaled_test.csv"

# =========================
# LOAD DATA
# =========================
train_data = pd.DataFrame()

train_data = pd.concat([train_data,pd.read_csv(PATH / f"idiq{TRAIN_FILE}").drop(columns="Unnamed: 0")],axis=1)
train_data["speed"] = pd.read_csv(PATH / f"speed{TRAIN_FILE}")["N"]
train_data = pd.concat([train_data,pd.read_csv(PATH / f"xgeom{TRAIN_FILE}").drop(columns="Unnamed: 0")],axis=1)
train_data["hysteresis"] = pd.read_csv(PATH / f"hysteresis{TRAIN_FILE}")["total"]
train_data["joule"]      = pd.read_csv(PATH / f"joule{TRAIN_FILE}")["total"]

test_data = pd.DataFrame()

test_data = pd.concat([test_data,pd.read_csv(PATH / f"idiq{TEST_FILE}").drop(columns="Unnamed: 0")],axis=1)
test_data["speed"] = pd.read_csv(PATH / f"speed{TEST_FILE}")["N"]
test_data = pd.concat([test_data,pd.read_csv(PATH / f"xgeom{TEST_FILE}").drop(columns="Unnamed: 0")],axis=1)
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

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# =========================
# REGISTER
# =========================
def register_csv(contents, info):
    new_row = pd.DataFrame([contents], columns = info.columns)
    info = pd.concat([info, new_row])
    BASE_DIR = Path(__file__).resolve().parent
    SAVE_PATH = BASE_DIR / ".." / "results_patu" / f"{MOTOR}" / f"motor_{MOTOR}_{var}_info.csv"
    info.to_csv(SAVE_PATH, index=False)
    return info

columns = ['neurons', 'layers', 'learn_rate', 'epochs', f'{var}_score', f'{var}_mse', f'{var}_mape', 'time'] 
info = pd.DataFrame(columns = columns)

# =========================
# CONFIG
# =========================
neurons = np.arange(10, 201, 10)
layers = [1, 2]
learning_rates = [0.1, 0.01]
epochs = 100
BATCH_SIZE = 256

# =========================
# DATASETS
# =========================
train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

train_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# GLOBAL BEST
# =========================
best_mape = float("inf")
best_model_block = None

best_global_mape = float("inf")
best_curve_global = None

# =========================
# MAIN LOOP
# =========================

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            
            print("============")
            print(f"\nTraining model --- neurons: {neurons[i]} -layers: {layers[j]} -lr: {learning_rates[k]}")
            print("============")
            print("")

            input_dim = len(train_data.columns.drop(target))
            model = RegressionModel(input_dim, 1, neurons[i], layers[j])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[k])
            start_time = datetime.datetime.now()

            best_mape_so_far = []
            best_mape = float("inf")

            # ===== TRAIN POR EPOCH =====
            for ep in range(epochs):

                print(f"======= Epoca {ep} =======")

                model.train()
                for X, y in train_loader_full:
                    X, y = X.to(device), y.to(device)

                    pred_train = model(X)
                    loss = loss_func(pred_train, y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    

                # ===== EVAL A CADA EPOCH =====
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

                Jou_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
                Jou_mse = mean_squared_error(y_test.detach().numpy(), y_pred.detach().numpy())
                Jou_mape = mean_absolute_percentage_error(y_test.numpy(), y_pred.numpy())

                # 🔥 BEST MAPE ATÉ A ÉPOCA
                if Jou_mape < best_mape:
                    best_mape = Jou_mape

                best_mape_so_far.append(Jou_mape)

                print(f"best_mape = {best_mape} || Jou_mape = {Jou_mape}")
                print("")
            
            if best_mape < best_global_mape:
                best_global_mape = best_mape
                best_curve_global = best_mape_so_far.copy()
                best_model_block = model.linear

            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()    
            contents = [neurons[i], layers[j], learning_rates[k], epochs, Jou_score, Jou_mse, Jou_mape, elapsed_time]
            info = register_csv(contents, info)

# =========================
# SAVE CURVE
# =========================
curve_df = pd.DataFrame({
    "epoch": np.arange(1, epochs + 1),
    "best_mape": best_curve_global
})

SAVE_CURVE = BASE_DIR.parent / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_epochs_{MOTOR}_{var}.csv"
SAVE_CURVE.parent.mkdir(parents=True, exist_ok=True)
curve_df.to_csv(SAVE_CURVE, index=False)
print("\nCurva salva em:", SAVE_CURVE)

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(curve_df["epoch"], curve_df["best_mape"], label="Baseline")
plt.xlabel("Epoch")
plt.ylabel("Best MAPE")
plt.title("Baseline — Best MAPE vs Epochs")
plt.grid(True)
plt.legend()
plt.show()

# =========================
# SAVE BEST WEIGHTS
# =========================
SAVE_DIR = BASE_DIR.parent / "transferLearning" / "data_pesos"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PATH = SAVE_DIR / f"pesos_V_{var}.pt"
torch.save(best_model_block, SAVE_PATH)

print("the end")