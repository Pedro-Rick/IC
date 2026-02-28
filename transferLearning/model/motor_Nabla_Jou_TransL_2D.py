import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

torch.manual_seed(42)
np.random.seed(42)

MOTOR = "Nabla"
MOTOR_TL = "2D"
var = "Jou"
target = ["joule"]

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
PATH = ROOT_DIR / "dataset" / MOTOR

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
# MODEL TL
# =========================
class TransLRegressionModel(nn.Module):

    def __init__(self, input_dim, peso_path):
        super().__init__()

        full_model = torch.load(
            peso_path,
            map_location="cpu",
            weights_only=False
        )

        self.pretrained_block = full_model

        first_linear = self.pretrained_block[0]
        pre_input_dim = first_linear.in_features

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, pre_input_dim),
            nn.ReLU()
        )

        for p in self.pretrained_block.parameters():
            p.requires_grad = False

        layers = list(self.pretrained_block.children())

        for layer in layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.adapter(x)
        x = self.pretrained_block(x)
        return x

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

def register_csv(contents, info):
    new_row = pd.DataFrame([contents], columns=info.columns)
    info = pd.concat([info, new_row])

    SAVE_PATH = BASE_DIR / ".." / "transL_results" / f"{MOTOR}"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_PATH / f"motor_{MOTOR}_{var}_TransL_{MOTOR_TL}_info.csv"

    info.to_csv(SAVE_PATH, index=False)
    return info

# =========================
# DATA LOADERS
# =========================
BATCH_SIZE = 256

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# PESOS
# =========================
arquivo = BASE_DIR / ".." / "data_pesos" / f"pesos_{MOTOR_TL}_{var}.pt"

# =========================
# CONFIG TL
# =========================
ft_learning_rates = [1e-2, 1e-3, 5e-4, 1e-4]
epochs = 100

columns = ["lr", "epochs", f"{var}_score", f"{var}_mse", f"{var}_mape", "time"]
info = pd.DataFrame(columns=columns)

fractions = [0.01, 0.05, 0.1, 0.25, 1.0]
seeds = [0, 1, 2, 3, 4]
curve_results = []

full_indices = np.arange(len(train_dataset))

# =========================
# MAIN LOOP
# =========================
for frac in fractions:

    print("\n====================")
    print("FRACTION =", frac)
    print("====================")
    
    mape_runs = []

    for seed in seeds:

        print(f"\n--- SEED {seed} ---")

        rng = np.random.default_rng(seed)

        subset_size = int(len(train_dataset) * frac)
        subset_idx = rng.choice(full_indices, subset_size, replace=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(subset_idx)
        )

        best_mape_seed = float("inf")

        for lr in ft_learning_rates:

            print(f"\nTraining model --- {lr}-{epochs}\n")

            model = TransLRegressionModel(
                input_dim=len(train_data.columns.drop(target)),
                peso_path=arquivo
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr
            )
            start_time = datetime.datetime.now()

            # ===== TREINO =====
            for _ in range(epochs):
                model.train()
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    pred_train = model(X)
                    loss = loss_func(pred_train, y)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # ===== TESTE =====
            time = datetime.datetime.now()

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

            print(f"R2={Jou_score:.4f} | MSE={Jou_mse:.4e} | MAPE={Jou_mape:.4f}")

            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds() 
            contents = [lr, epochs, Jou_score, Jou_mse, Jou_mape, elapsed_time]
            info = register_csv(contents, info)

            if Jou_mape < best_mape_seed:
                best_mape_seed = Jou_mape

        mape_runs.append(best_mape_seed)

    # ===== CURVA TL (BEST) =====
    curve_results.append({
        "fraction": frac,
        "best_mape_mean": np.mean(mape_runs),
        "best_mape_std": np.std(mape_runs),
        "best_mape": np.min(mape_runs)
    })

# =========================
# SAVE CURVE TL
# =========================
curve_df = pd.DataFrame(curve_results)

curve_path = BASE_DIR / ".." / "transL_results" / f"{MOTOR}" / "graficos" / f"curve_TL_{MOTOR}_{var}.csv"
curve_path.parent.mkdir(parents=True, exist_ok=True)
curve_df.to_csv(curve_path, index=False)

print("Curva TL salva em:", curve_path)

# =========================
# LOAD BASELINE (POR FRACTION)
# =========================
baseline_path = BASE_DIR / ".." / ".." / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_baseline_{MOTOR}_{var}.csv"
base_df = pd.read_csv(baseline_path)

# =========================
# PLOT COMPARAÇÃO
# =========================
plt.figure()

plt.plot(
    base_df["fraction"],
    base_df["best_mape_mean"],
    'o-',
    label="Baseline"
)

plt.plot(
    curve_df["fraction"],
    curve_df["best_mape_mean"],
    's--',
    label="Transfer Learning"
)

plt.fill_between(
    base_df["fraction"],
    base_df["best_mape_mean"] - base_df["best_mape_std"],
    base_df["best_mape_mean"] + base_df["best_mape_std"],
    alpha=0.2
)

plt.xscale("log")
plt.xlabel("Fração dos dados de treino")
plt.ylabel("Best MAPE")
plt.title(f"{MOTOR} — Baseline vs TL (Best)")
plt.legend()
plt.grid(True)

save_fig = BASE_DIR / ".." / "transL_results" / f"{MOTOR}" / "graficos"
save_fig.mkdir(parents=True, exist_ok=True)

plt.savefig(save_fig / f"baseline_{MOTOR}_vs_{MOTOR}_{var}_TL_{MOTOR_TL}.png")
plt.show()

print("\nFIM")