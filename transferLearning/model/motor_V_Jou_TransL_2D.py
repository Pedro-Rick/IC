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

MOTOR = "v"
MOTOR_TL = "2D"
var = "Jou"
target = ["joule"]

BASE_DIR = Path(__file__).resolve().parent

PATH = BASE_DIR.parent / "dataset" / MOTOR

TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE  = "_all_scaled_test.csv"

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


class TransLRegressionModel(nn.Module):

    def __init__(self, input_dim, peso_path):
        super().__init__()

        # carregar modelo completo salvo
        full_model = torch.load(
            peso_path,
            map_location="cpu",
            weights_only=False
        )

        # backbone pré-treinado
        self.pretrained_block = full_model

        # input esperado pelo modelo V
        first_linear = self.pretrained_block[0]
        pre_input_dim = first_linear.in_features

        # adapter das entradas
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, pre_input_dim),
            nn.ReLU()
        )

        # CONGELA TUDO
        for p in self.pretrained_block.parameters():
            p.requires_grad = False

        # LIBERA ÚLTIMAS 2 CAMADAS
        layers = list(self.pretrained_block.children())

        for layer in layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
    def forward(self, x):
        x = self.adapter(x)
        x = self.pretrained_block(x)
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

    SAVE_PATH = BASE_DIR / ".." / "transL_results" / f"{MOTOR}"
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = SAVE_PATH / f"motor_{MOTOR}_{var}_TransL_{MOTOR_TL}_info.csv"

    info.to_csv(SAVE_PATH, index=False)
    return info

BATCH_SIZE = 256

train_dataset = MotorDataset(train_data.drop(columns=target), train_data[target])
test_dataset  = MotorDataset(test_data.drop(columns=target), test_data[target])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# buscando o arquivo
arquivo = BASE_DIR / ".." / "data_pesos" / f"pesos_{MOTOR_TL}_{var}.pt"

#para finetuning

ft_learning_rates = [1e-3, 5e-4]
epochs = 100

columns = ["lr", "epochs", f"{var}_score", f"{var}_mse", f"{var}_mape", "time"]

info = pd.DataFrame(columns=columns)

# definindo a variaveis do grafico
fractions = [0.01, 0.05, 0.1, 0.25, 1.0]
curve_results = []

full_indices = np.arange(len(train_dataset))

for frac in fractions:

    print("\n====================")
    print("FRACTION =", frac)
    print("====================")

    subset_size = int(len(train_dataset) * frac)
    subset_idx = np.random.choice(full_indices, subset_size, replace=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(subset_idx)
    )

    best_mape_frac = float("inf")

for i in range(len(ft_learning_rates)):

    print(f"\nTraining model --- {ft_learning_rates[i]}-{epochs}\n")

    model = TransLRegressionModel(
        input_dim = len(train_data.columns.drop(target)),
        peso_path= arquivo 
    )

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=ft_learning_rates[i])

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

    Jou_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
    Jou_mse = mean_squared_error(y_test.detach().numpy(), y_pred.detach().numpy())
    Jou_mape = mean_absolute_percentage_error(y_test.detach().numpy(), y_pred.detach().numpy())


    print(f"R2={Jou_score:.4f} | MSE={Jou_mse:.4e} | MAPE={Jou_mape:.4f}")

    contents = [ft_learning_rates[i], epochs, Jou_score, Jou_mse, Jou_mape, time]

    info = register_csv(contents, info)

    if Jou_mape < best_mape_frac:
        best_mape_frac = Jou_mape

# salvando a curva de resultados
curve_results.append({
    "fraction": frac,
    "best_mape": best_mape_frac
})

print("BEST TL MAPE =", best_mape_frac)

curve_df = pd.DataFrame(curve_results)

curve_path = BASE_DIR / "results_patu" / f"{MOTOR}" / "graficos" / f"curve_tl_{MOTOR}_{var}.csv"
curve_df.to_csv(curve_path, index=False)

print("Curva TL salva em:", curve_path)

baseline_path = BASE_DIR / "transferLearning" / "tranL_results" / {MOTOR} / "graficos" / f"curve_baseline_{MOTOR}_{var}.csv"
base_df = pd.read_csv(baseline_path)

plt.figure()

plt.plot(
    base_df["fraction"],
    base_df["best_mape"],
    'o-',
    label="Baseline"
)

plt.plot(
    curve_df["fraction"],
    curve_df["best_mape"],
    's--',
    label="Transfer Learning"
)

plt.xscale("log")
plt.xlabel("Fração dos dados de treino")
plt.ylabel("Melhor MAPE")
plt.title(f"{MOTOR} — Baseline vs TL")
plt.legend()
plt.grid(True)

plt.savefig(BASE_DIR / f"compare_TL_vs_base_{MOTOR}_{var}_TL_{MOTOR_TL}.png")
plt.show()

print("\nFIM")