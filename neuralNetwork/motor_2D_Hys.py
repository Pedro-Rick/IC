import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset, SubsetRandomSampler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# data loading

MOTOR = "2D"
PATH = f"../dataset/{MOTOR}/"
TRAIN_FILE = "_all_scaled_train.csv"
TEST_FILE = "_all_scaled_test.csv"
 
train_data = pd.DataFrame()

train_data = pd.concat([train_data, pd.read_csv(f'{PATH}idiq{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
train_data['speed'] = pd.read_csv(f'{PATH}speed{TRAIN_FILE}')['N']
train_data = pd.concat([train_data, pd.read_csv(f'{PATH}xgeom{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
train_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TRAIN_FILE}')['total']
train_data['joule'] = pd.read_csv(f'{PATH}joule{TRAIN_FILE}')['total']

test_data = pd.DataFrame()

test_data = pd.concat([test_data, pd.read_csv(f'{PATH}idiq{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
test_data['speed'] = pd.read_csv(f'{PATH}speed{TEST_FILE}')['N']
test_data = pd.concat([test_data, pd.read_csv(f'{PATH}xgeom{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
test_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TEST_FILE}')['total']
test_data['joule'] = pd.read_csv(f'{PATH}joule{TEST_FILE}')['total']


class RegressionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, neurons = 5, layers = 1):
        super().__init__()

        modules = []
        
        modules.append(nn.Linear(input_dim, neurons))
        modules.append(nn.ReLU())
        for i in range(layers):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(neurons, output_dim))
        
        self.linear = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.linear(x)
        return x

class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def register_csv(contents, info):
    new_row = pd.DataFrame([contents], columns = info.columns)
    info = pd.concat([info, new_row])
    info.to_csv(r'../results_patu/2D/motor_2D_Hys_info.csv')
    return info

target = ['hysteresis']

neurons = np.arange(10, 200 + 1, 10)
layers = [1, 2]
learning_rates = [0.1, 0.01]
epochs = 100

train_dataset = MotorDataset(train_data.drop(columns = target), train_data[target])
test_dataset = MotorDataset(test_data.drop(columns = target), test_data[target])

BATCH_SIZE = 256

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

columns = ['neurons', 'layers', 'learn_rate', 'epochs', 'hys_score', 'hys_mse', 'hys_mape', 'time']
info = pd.DataFrame(columns = columns)

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            print(f"\nTraining model --- {neurons[i]}-{layers[j]}-{learning_rates[k]}-{epochs}\n")
            
            input_dim = len(train_data.columns.drop(target))
            output_dim = 1
            
            model = RegressionModel(input_dim, output_dim, neurons[i], layers[j])
            
            loss_func = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rates[k])

            for a in range(epochs):
                model.train()
                for X, y in train_loader:
                    pred_train = model(X)
                    loss = loss_func(pred_train, y)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            time = datetime.datetime.now()

            print(f"\tFinished training model at {time}.\n")

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

            print(f"\tSpecs:")
            print(f"\t\thys_score: {hys_score}, hys_mse: {hys_mse}, hys_mape: {hys_mape}.\n")

            contents = [neurons[i], layers[j], learning_rates[k], epochs, hys_score, hys_mse, hys_mape, time] 

            info = register_csv(contents, info)
            
print(f"the end")