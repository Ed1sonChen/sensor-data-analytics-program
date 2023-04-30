#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2023/2/14
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
The sensor data acquires a data every 20 points, plus H+R+S+D four-dimensional data, a total of 54-dimensional data is used as the input of the model, and S+D is used as the fitting target of the model.
The training data and test data are standardized, and then the training set is used for model training, and the test set is used for model verification.

model
The first layer builds a bidirectional lstm model with 2 layers, followed by activation and deactivation functions
The second layer is a linear function, followed by an activation function and a deactivation function;
The third layer is a linear function, followed by an activation function
'''

def load_data(seed=98):
    data_set_train = np.load("/content/drive/MyDrive/sim_data/simu_20000_0.1_90_140_train.npy")
    x1 = data_set_train[..., 0:1000:20]
    x2 = data_set_train[..., 1002:]

    data_set_test = np.load("/content/drive/MyDrive/sim_data/simu_10000_0.1_141_178_test.npy")
    x3 = data_set_test[..., 0:1000:20]
    x4 = data_set_test[..., 1002:]

    X_train = np.concatenate([x1, x2], axis=1)
    X_test = np.concatenate([x3, x4], axis=1)
    X_train = X_train[:, :]
    X_test = X_test[:, :]
    y_test = data_set_test[:, -2:]
    y_train = data_set_train[:, -2:]

    SCALER = MinMaxScaler()
    SCALER.fit(X_train)
    X_train = SCALER.transform(X_train)
    X_test = SCALER.transform(X_test)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    # Here, the training set can be divided into training set and verification set according to 90% and 10%. For faster training, there is no division
    # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.09, random_state=seed,
    #                                                     shuffle=True)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_eval.shape, y_eval.shape)
    return X_train, X_test, y_train, y_test


class Model(torch.nn.Module):

    def __init__(self, inputs_size, hidden_size, outputs_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.GRU(input_size=inputs_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, outputs_size)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.activation = torch.nn.ReLU()
        self.bc = torch.nn.BatchNorm1d(hidden_size)

    def attention_forward(self, x, query):
        d_k = query.shape[-1]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        outputs = torch.matmul(alpha_n, x).sum(dim=1)
        return outputs, alpha_n

    def lstm_forward(self, inputs):
        outputs, (h_n, c_n) = self.lstm(inputs)
        outputs, _ = self.attention_forward(outputs, self.dropout1(outputs))
        return outputs

    def forward(self, x):
        output = self.lstm_forward(x)
        output = output.reshape(-1, self.hidden_size)
        output = self.bc(output)
        output = self.activation(output)
        output = self.dropout2(output)
        out = self.linear(output)
        return out


def train(seed=100):
    epoches = 2000
    hidden_size = 2048
    n_components = 54
    out_size = 2
    lr = 0.01
    batch_size = 2048
    X_train, X_test, y_train, y_test = load_data(seed=seed)  # load data
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())  # trainning dataset
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())  # testing dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    least_loss = 10000
    loss_counter = 0
    net = Model(n_components, hidden_size, out_size).to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    history = []
    for epoch in range(epoches):
        total_mae = 0
        total_loss = 0
        number = 0
        for index, (batch_x, batch_y) in enumerate(train_loader):
            out = net(batch_x.to(DEVICE))
            loss = criterion(out, batch_y.to(DEVICE))
            optim.zero_grad()
            loss.backward()
            mae = mean_absolute_error(out.detach().cpu(), batch_y)
            total_mae += mae
            l = loss.item()
            total_loss += l
            optim.step()
            number += 1
        history.append([total_mae / number, total_loss / number])
        net.eval()
        eval_mae = 0
        counter = 0
        # In order to observe the fitting effect and ensure that the model does overfit, the test set is used to observe the fitting effect, and the early stopping mechanism of loss_counter is used to control the fitting effect of the model
        for index, (batch_x, batch_y) in enumerate(test_loader):
            out = net(batch_x.to(DEVICE))
            mae = np.mean(abs(np.array(out.detach().cpu()) - np.asarray(batch_y)))
            eval_mae += mae
            counter += 1
        print("mae:", epoch, loss_counter, total_mae / number, eval_mae / counter)
        net.train()
        if least_loss > eval_mae / counter:
            least_loss = eval_mae / counter
            torch.save(net, "best.pt", )
            loss_counter = 0
        else:
            loss_counter += 1
        if loss_counter > 200:
            break
    net = torch.load("best.pt")
    net.eval()
    result = []
    for index, (batch_x, batch_y) in enumerate(test_loader):
        out = net(batch_x.to(DEVICE))
        out = out.detach().cpu().numpy().squeeze()
        result.extend(out)
        # loss = mean_absolute_error(np.asarray(out), np.asarray(batch_y))
    result = np.asarray(result)
    np.save('data.npy', result)


if __name__ == '__main__':
    train(seed=18)
