#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 2023/2/14
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(seed=98):
    data_set_train = np.load("simu_20000_0.1_90_140_train.npy")
    x1 = data_set_train[..., 0:1000:20]
    x2 = data_set_train[..., 1002:]

    data_set_eval = np.load("simu_10000_0.1_141_178_test.npy")
    x3 = data_set_eval[..., 0:1000:20]
    x4 = data_set_eval[..., 1002:]

    X_train = np.concatenate([x1, x2], axis=1)
    X_eval = np.concatenate([x3, x4], axis=1)
    y_eval = data_set_eval[:, -2:]
    y_train = data_set_train[:, -2:]

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # Fit StandardScaler to the training set only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform the training, validation, and evaluation sets using the scaler fit to the training set
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_eval = scaler.transform(X_eval)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_eval = np.expand_dims(X_eval, axis=1)

    return X_train, X_val, y_train, y_val, X_eval, y_eval


class Model(torch.nn.Module):
    def __init__(self, inputs_size, hidden_size, outputs_size):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(inputs_size, hidden_size, num_layers=3, batch_first=True, dropout=0.1,
                                  bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.predict = torch.nn.Linear(hidden_size // 2, outputs_size)
        self.activation = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.05)

    def lstm_forward(self, inputs):  # torch.Size([B, 30, 5])
        outputs, (h_n, c_n) = self.lstm(inputs)  # torch.Size([B, 30, 128])
        return outputs[:, -1, :]  # torch.Size([B, 128])

    def forward(self, inputs):  # torch.Size([B, 30, 5])
        outputs = self.lstm_forward(inputs)  # torch.Size([B, 128])
        outputs = self.activation(outputs)  # torch.Size([B, 128])
        outputs = self.dropout(outputs)  # torch.Size([B, 128])

        outputs = self.fc1(outputs)  # torch.Size([B, 64])
        outputs = self.activation(outputs)  # torch.Size([B, 64])
        outputs = self.dropout(outputs)  # torch.Size([B, 64])

        outputs = self.fc2(outputs)  # torch.Size([B, 32])
        outputs = self.activation(outputs)  # torch.Size([B, 32])
        outputs = self.dropout(outputs)  # torch.Size([B, 32])

        outputs = self.predict(outputs)  # torch.Size([B, 4])

        return outputs


def train(mod="train", seed=100):
    epoches = 4000
    hidden_size = 2048
    n_components = 54
    out_size = 2
    lr = 0.0001
    batch_size = 512
    X_train, X_test, y_train, y_test, X_eval, y_eval = load_data(seed=seed)  # read data
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())  # trainning 
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())  # testing
    eval_dataset = TensorDataset(torch.tensor(X_eval).float(), torch.tensor(y_eval).float())  # testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    least_loss = 10000
    loss_counter = 0
    net = Model(n_components, hidden_size, out_size).to(DEVICE)
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
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
        for index, (batch_x, batch_y) in enumerate(eval_loader):
            out = net(batch_x.to(DEVICE))
            mae = np.mean(abs(np.array(out.detach().cpu()) - np.asarray(batch_y)))
            eval_mae += mae
            counter += 1
        print("mae:", epoch, eval_mae / counter, total_mae / number)
        net.train()
        if least_loss > eval_mae / counter:
            least_loss = eval_mae / counter
            torch.save(net, "best.pt", )
            loss_counter = 0
        else:
            loss_counter += 1
        if loss_counter > 15:
            break
    net = torch.load("best.pt")
    net.eval()
    result = []
    for index, (batch_x, batch_y) in enumerate(eval_loader):
        out = net(batch_x.to(DEVICE))
        out = out.detach().cpu().numpy().squeeze()
        result.extend(out)
        loss = mean_absolute_error(np.asarray(out), np.asarray(batch_y))
        print(index, loss, out, batch_y)
    result = np.asarray(result)
    np.save('data.npy', result)


if __name__ == '__main__':
    train(mod="train", seed=18)
