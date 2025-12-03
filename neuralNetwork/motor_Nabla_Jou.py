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
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# data loading

train_data = pd.DataFrame()

train_data['hysteresis'] = pd.read_csv(r'../dataset/Nabla/hysteresis_all_scaled_train.csv')['total']
train_data['id'] = pd.read_csv(r'../dataset/Nabla/idiq_all_scaled_train.csv')['id']
train_data['iq'] = pd.read_csv(r'../dataset/Nabla/idiq_all_scaled_train.csv')['iq']
train_data['joule'] = pd.read_csv(r'../dataset/Nabla/joule_all_scaled_train.csv')['total']
train_data['speed'] = pd.read_csv(r'../dataset/Nabla/speed_all_scaled_train.csv')['N']
train_data['d1'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d1']
train_data['d2'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d2']
train_data['d3'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d3']
train_data['d4'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d4']
train_data['d5'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d5']
train_data['d6'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d6']
train_data['d7'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d7']
train_data['d8'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_train.csv')['d8']


test_data = pd.DataFrame()

test_data['hysteresis'] = pd.read_csv(r'../dataset/Nabla/hysteresis_all_scaled_test.csv')['total']
test_data['id'] = pd.read_csv(r'../dataset/Nabla/idiq_all_scaled_test.csv')['id']
test_data['iq'] = pd.read_csv(r'../dataset/Nabla/idiq_all_scaled_test.csv')['iq']
test_data['joule'] = pd.read_csv(r'../dataset/Nabla/joule_all_scaled_test.csv')['total']
test_data['speed'] = pd.read_csv(r'../dataset/Nabla/speed_all_scaled_test.csv')['N']
test_data['d1'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d1']
test_data['d2'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d2']
test_data['d3'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d3']
test_data['d4'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d4']
test_data['d5'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d5']
test_data['d6'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d6']
test_data['d7'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d7']
test_data['d8'] = pd.read_csv(r'../dataset/Nabla/xgeom_all_scaled_test.csv')['d8']


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

def register_csv(contents, info):
    new_row = pd.DataFrame([contents], columns = info.columns)
    info = pd.concat([info, new_row])
    info.to_csv(r'../results_patu/Nabla/motor_Nabla_Jou_info.csv') # mudar Jou e Hys
    return info

target = ['hysteresis', 'joule']

variable = 'joule' # mudar joule e hysteresis

neurons = np.arange(10, 200 + 1, 10)
layers = [1, 2]
learning_rates = [0.1, 0.01]
epochs = 100

X_train = torch.tensor(train_data.drop(columns = target).values, dtype=torch.float32)
y_train = torch.tensor(train_data[variable].values, dtype=torch.float32)
X_test = torch.tensor(test_data.drop(columns = target).values, dtype=torch.float32)
y_test = torch.tensor(test_data[variable].values, dtype=torch.float32)

columns = ['neurons', 'layers', 'learn_rate', 'epochs', 'jou_score', 'jou_mse', 'jou_mape', 'time'] # mudar Jou e Hys
info = pd.DataFrame(columns = columns)

for i in range(len(neurons)):
    for j in range(len(layers)):
        for k in range(len(learning_rates)):
            print(f"\nTraining model --- {neurons[i]}-{layers[j]}-{learning_rates[k]}-{epochs}\n")
            
            input_dim = len(train_data.columns.drop(target))
            # output_dim = len(target)
            output_dim = 1
            
            model = RegressionModel(input_dim, output_dim, neurons[i], layers[j])
            
            loss_func = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rates[k])
            
            losses = torch.zeros(epochs)

            for a in range(epochs):
                pred = model(X_train)
                pred = pred.squeeze()
            
                loss = loss_func(pred, y_train)
                losses[a] = loss
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            time = datetime.datetime.now()
            y_pred = model(X_test)

            print(f"\tFinished training model at {time}.\n")

            jou_score = r2_score(y_test.detach().numpy(), y_pred.detach().numpy())
            jou_mse = mean_squared_error(y_test.detach().numpy(), y_pred.detach().numpy())
            jou_mape = mean_absolute_percentage_error(y_test.detach().numpy(), y_pred.detach().numpy())

            print(f"\tSpecs:")
            print(f"\t\tjou_score: {jou_score}, jou_mse: {jou_mse}, jou_mape: {jou_mape}.\n\n")

            contents = [neurons[i], layers[j], learning_rates[k], epochs, jou_score, jou_mse, jou_mape, time] # mudar Jou e Hys
            
            info = register_csv(contents, info)
            # register_txt(contents, info)
print(f"the end")