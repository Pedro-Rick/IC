import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

MOTOR = "V"
MOTOR_TL = "Nabla"
var = "Hys"
target = ["hysteresis"]

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

test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = len(train_data.columns.drop(target))

ARQ_PESOS = IC_BASE_DIR / "transferLearning" / "data_pesos" / f"pesos_{MOTOR_TL}_{var}.pt"

# =========================
# PARAMETERS
# =========================
b_TL_lr = float(get_best_mape_row("learn_rate"))
b_TL_neurons = int(get_best_mape_row("max_neurons"))
b_TL_layers = int(get_best_mape_row("layers"))

results_curves = {}
mape_seeds = []

columns = ['seed', 'max_neurons', 'layers', 'lr',  "epoch", f'{var}_score', f'{var}_mse', f'{var}_rmse', f'{var}_mape'] 
info = pd.DataFrame(columns = columns)
columns_percent = ["seed", "max_neurons", "layers", "lr", "epoch", "n_samples", "data_fraction", f"{var}_score", f"{var}_mse", f"{var}_rmse", f"{var}_mape"]
info_percent = pd.DataFrame(columns=columns_percent)

SAVE_CONTS =  IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / f"{MOTOR}_TL_arq_{MOTOR_TL}_{var}_info.csv"
SAVE_CONTS_PERCENT = IC_BASE_DIR / "transferLearning" / "TL_results"/ f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / f"percent_data_{MOTOR}_TL_arq_{MOTOR_TL}_{var}_info.csv"

# =========================
# CONFIG
# =========================
epochs = 100
seeds = 30
fractions = [0.1, 0.25, 0.5, 1]

# =========================
# MAIN LOOP
# =========================
for frac in fractions:

    print(f"\n===== FRACTION {int(frac*100)}% =====\n")

    for seed in range (seeds):

        print("")
        print(f"===== Seed {seed + 1} =====")
        print("")

        # embaralha
        train_shuffled = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        # subset
        n_samples = int(len(train_shuffled) * frac)
        train_subset = train_data.sample(frac=frac, random_state=seed)

        train_dataset = MotorDataset(train_subset.drop(columns=target), train_subset[target])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = RegressionModel(input_dim=input_dim, output_dim=1, neurons=b_TL_neurons, layers=b_TL_layers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=b_TL_lr)

        mape_epochs = []

        for ep in range(epochs):

            model.train()

            for X,y in train_loader:

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
            
            Hys_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
            Hys_mse=mean_squared_error(y_test.numpy(),y_pred.numpy())
            Hys_rmse=np.sqrt(Hys_mse)
            Hys_mape=mean_absolute_percentage_error(y_test.numpy(),y_pred.numpy())
            
            if (frac == 1):
                mape_epochs.append(Hys_mape)

                contents = [(seed+1), b_TL_neurons, b_TL_layers, b_TL_lr, (ep+1), Hys_score, Hys_mse, Hys_rmse, Hys_mape]
                info = register_csv(contents, info, SAVE_CONTS)

                if ((((ep+1)%50)== 0) or ((ep) == 0)):
                    print(f"Epoch {ep +1} || MAPE = {Hys_mape}")

        if (frac != 1):
            print(f"SCORE = {Hys_score} || MAPE = {Hys_mape}") 
        
        contents_fraction = [(seed+1), b_TL_neurons, b_TL_layers, b_TL_lr, (ep+1), n_samples, frac, Hys_score, Hys_mse, Hys_rmse, Hys_mape]
        info_percent = register_csv(contents_fraction, info_percent, SAVE_CONTS_PERCENT)
        
        if (frac == 1):
            mape_seeds.append(mape_epochs)

# =========================
# CALCULA MÉDIA E STD
# =========================
mape_seeds = np.array(mape_seeds)

mape_mean = mape_seeds.mean(axis=0)
mape_std  = mape_seeds.std(axis=0)

score_mape = mape_mean[-1]

print("")
print("MAPE médio final:",score_mape)

results_curves["TL_arquitetura"] = (mape_mean, mape_std)


plt.figure()
# ========================= 
# LOAD BASELINE 
# =========================
baseline_path = IC_BASE_DIR / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_epochs_{MOTOR}_{var}_MAPE.csv"

if baseline_path.exists():
    base_df = pd.read_csv(baseline_path)

    line, = plt.plot(
        base_df["epoch"],
        base_df["mape_mean"],
        label=f"baseline"
    )

    plt.fill_between(
        base_df["epoch"],
        base_df["mape_mean"] - base_df["mape_std"],
        base_df["mape_mean"] + base_df["mape_std"],
        color=line.get_color(),
        alpha=0.25
    )
else:
    print(f"CSV curva baseline MAPE ainda não existe")

# ========================= 
# LOAD TL PESOS 
# =========================
TL_PESOS_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / "graficos" / f"curve_TL_pesos_{MOTOR}_{var}.csv"

if TL_PESOS_path.exists():
    pesos_df = pd.read_csv(TL_PESOS_path)

    line, = plt.plot(
        pesos_df["epoch"],
        pesos_df["mape_mean"],
        label=f"TL_pesos"
    )

    plt.fill_between(
        pesos_df["epoch"],
        pesos_df["mape_mean"] - pesos_df["mape_std"],
        pesos_df["mape_mean"] + pesos_df["mape_std"],
        color=line.get_color(),
        alpha=0.25
    )

else:
    print(f"CSV curva pesos MAPE ainda não existe")

# =========================
# SAVE + PLOT CURVES
# =========================
for name,(curve_mean,curve_std) in results_curves.items():

    curve_df = pd.DataFrame({
        "epoch": np.arange(1, epochs + 1),
        "mape_mean": curve_mean,
        "mape_std": curve_std
    })

    curve_path = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / "graficos" / f"curve_{name}_{MOTOR}_{var}.csv"

    curve_path.parent.mkdir(parents=True,exist_ok=True)

    curve_df.to_csv(curve_path,index=False)

    print("Curva TL salva em:",curve_path)

    line, = plt.plot(
        curve_df["epoch"],
        curve_df["mape_mean"],
        label=name
    )

    plt.fill_between(
        curve_df["epoch"],
        curve_df["mape_mean"] - curve_df["mape_std"],
        curve_df["mape_mean"] + curve_df["mape_std"],
        color=line.get_color(),
        alpha=0.25
    )

plt.xlabel("Epoch")
plt.ylabel("MAPE")

plt.title(f"{MOTOR}_{var} - TL:{MOTOR_TL} - TLa, TLap, Baseline - MAPE ")

plt.grid(True)
plt.legend()

save_fig = IC_BASE_DIR / "transferLearning" / "TL_results" / f"{MOTOR}" / f"{MOTOR}_TL_{MOTOR_TL}" / "graficos"

save_fig.mkdir(parents=True,exist_ok=True)

plt.savefig(save_fig / f"baseline_TLa_TLp-{MOTOR}_TL_{MOTOR_TL}_{var}_MAPE.png")

plt.show()

print("\nFIM")