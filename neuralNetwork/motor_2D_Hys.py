import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "2D"
var = "Hys"
target = ['hysteresis']

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
def register_csv(contents, info, path):
    new_row = pd.DataFrame([contents], columns = info.columns)
    info = pd.concat([info, new_row])
    info.to_csv(path, index=False)
    return info

save_path = BASE_DIR / ".." / "results_patu" / f"{MOTOR}" / f"motor_{MOTOR}_{var}_info.csv"
columns = ['max_neurons', 'layers', 'learn_rate', 'epochs', f'{var}_score', f'{var}_mse', f'{var}_rmse', f'{var}_mape'] 
info = pd.DataFrame(columns = columns)

# =========================
# CONFIG
# =========================
neurons = np.arange(350, 400 + 1, 10)
layers = [1, 2, 5, 10]
learning_rates = [0.1, 0.01]
epochs = 100
BATCH_SIZE = 256

seeds = 30

results_curves = {}
best_mape = float("inf")
curve_mape = []
media_mape_epochs = []

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
best_model_mape = None

best_curve_global = None

# =========================
# MAIN LOOP
# =========================

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            
            print("============")
            print(f"\nTraining model --- neurons: {neurons[i]} -layers: {layers[j]} -lr: {learning_rates[k]}")
            print("")

            neuron_per_layer = int(neurons[i]/layers[j])

            input_dim = len(train_data.columns.drop(target))
            model = RegressionModel(input_dim, 1, neuron_per_layer, layers[j])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[k])

            # ===== TRAIN POR EPOCH =====
            for ep in range(epochs):

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

                Hys_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
                Hys_mse = mean_squared_error(y_test.detach().numpy(), y_pred.detach().numpy())
                Hys_rmse=np.sqrt(Hys_mse)
                Hys_mape = mean_absolute_percentage_error(y_test.numpy(), y_pred.numpy())
            
            if Hys_mape < best_mape:
                    best_mape = Hys_mape
                    best_model_mape = model.linear
                    b_neurons = neurons[i]
                    b_neurons_per_layer = int(neurons[i]/layers[j])
                    b_layers = layers[j]
                    b_lr = learning_rates[k]

            print(f"MAPE = {Hys_mape}")
            print(f"Best MAPE = {best_mape}")
            print("")

            contents = [neurons[i], layers[j], learning_rates[k], epochs, Hys_score, Hys_mse, Hys_rmse, Hys_mape]
            info = register_csv(contents, info, save_path)

# =========================
# GRÁFICO
# =========================
print("============")
print(f"\n Melhor modelo: neurons = {b_neurons} - layers = {b_layers} - lr = {b_lr}")
print("============")

columns = ['seed', 'max_neurons', 'layers', 'learn_rate',  "epoch", f'{var}_score', f'{var}_mse', f'{var}_rmse', f'{var}_mape'] 
info = pd.DataFrame(columns = columns)


for seed in range(seeds):

    print("")
    print(f"===== Seed {seed + 1} =====")
    print("")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RegressionModel(input_dim, 1, b_neurons_per_layer, b_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=b_lr)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mape_epochs = []

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

        Hys_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
        Hys_mse = mean_squared_error(y_test.numpy(),y_pred.numpy())
        Hys_rmse = np.sqrt(Hys_mse)
        Hys_mape = mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())

        mape_epochs.append(Hys_mape)

        contents = [seed, b_neurons, b_layers, b_lr, (ep+1), Hys_score, Hys_mse, Hys_rmse, Hys_mape]
        save = BASE_DIR.parent / "results_patu" / f"{MOTOR}" / f"motor_{MOTOR}_{var}_info_per_epochs.csv"
        info = register_csv(contents, info, save)

        if ((((ep+1)%50)== 0) or ((ep) == 0)):
            print(f"Epoch {ep +1} || MAPE = {Hys_mape}")

    media_mape_epochs.append(mape_epochs)


# =========================
# CALCULA MÉDIA E STD
# =========================

mape_array = np.array(media_mape_epochs)

mape_mean = np.mean(mape_array, axis=0)
mape_std  = np.std(mape_array, axis=0)

score_mape = mape_mean[-1]

print("")
print("MAPE médio final:",score_mape)

results_curves[f"Baseline"] = (mape_mean, mape_std)

# =========================
# PLOT
# =========================
plt.figure()

for name,(curve_mean,curve_std) in results_curves.items():

    curve_df = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "mape_mean": curve_mean,
        "mape_std": curve_std
    })

    line, = plt.plot(
            curve_df["epoch"],
            curve_df[f"MAPE_mean"],
            label=name
        )

    plt.fill_between(
        curve_df["epoch"],
        curve_df[f"MAPE_mean"] - curve_df[f"MAPE_std"],
        curve_df[f"MAPE_mean"] + curve_df[f"MAPE_std"],
        color=line.get_color(),
        alpha=0.25
    )



plt.xlabel("Epoch")
plt.ylabel(f"MAPE")
plt.title(f"{MOTOR}_{var} — Baseline")
plt.grid(True)
plt.legend()
plt.show()

# =========================
# SAVE CURVE
# =========================

SAVE_CURVE = BASE_DIR.parent / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_epochs_{MOTOR}_{var}_MAPE.csv"
SAVE_CURVE.parent.mkdir(parents=True, exist_ok=True)
curve_df.to_csv(SAVE_CURVE, index=False)
print("\nCurva salva em:", SAVE_CURVE)


# =========================
# SAVE BEST WEIGHTS
# =========================
SAVE_DIR = BASE_DIR.parent / "transferLearning" / "data_pesos"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PATH = SAVE_DIR / f"pesos_{MOTOR}_{var}.pt"
torch.save(best_model_mape, SAVE_PATH)

print("the end")