# Sensor-Data-Analytics-Program

## Description
A seismic sensor has been installed under a hospital bed and a family bed for monitoring vital signs of subjects on the bed. About 75 subjects have participated in the experiment. In each experiment, a subject has lied on one bed first for a few minutes, then lie on the other bed for a few minutes.

The goal of this project is to build a relationship model between the sensor data and the parameters (S, D), and predict S and D from future sensor data using Python language. The data file contains sensor data (100Hz * 10 seconds) + ID + Time + H + R + S + D, which may be visualized with the command "./view_data.py xxx.npy 6". Here, Time is the epoch time of the first sensor data sample and can be converted to time string with function epoch_time_local() in the Appendix. H, R, S and D are heart rate, respiratory rate, systolic and diastolic blood pressure.

The program should achieve a low MAE (less than 3 ideally) for two parameters (S, D) on the test data set and show a matching trend comparison plot of labels and prediction results (see main_plot.py as the example).

If you choose to perform supervised training, please use the train data set (_train.npy) for training and do not include the test data set (_test.npy) in training, which shall be used for test only.

---

## Quick Start

### Download
Some datasets require downloads in order to be used. All the files can be downloaded [here](https://www.dropbox.com/sh/kpiit4ly8l47mo4/AACqFLwGjgcOhyr6GN-669PZa?dl=0).

### Dependencies
The following dependencies are required to run this project:
* numpy
* pytorch
* sklearn

## Model
My model is a 3-layer model. 
The first layer constructs a bidirectional LSTM model with two layers, followed by activation and dropout functions.
The second layer is a linear function, followed by activation and dropout functions.
The third layer is a linear function, followed by an activation function.
```sh
class Model(torch.nn.Module):
    def __init__(self, inputs_size, hidden_size, outputs_size):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(inputs_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2,
                                  bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.predict = torch.nn.Linear(hidden_size, outputs_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def lstm_forward(self, inputs):  # torch.Size([B, 30, 5])
        outputs, (h_n, c_n) = self.lstm(inputs)  # torch.Size([B, 30, 128])
        return outputs[:, -1, :]  # torch.Size([B, 128])

    def forward(self, inputs):  # torch.Size([B, 30, 5])
        outputs = self.lstm_forward(inputs)  # torch.Size([B, 128])
        outputs = self.activation(outputs)  # torch.Size([B, 128])
        outputs = self.dropout(outputs)  # torch.Size([B, 128])

        outputs = self.fc(outputs)  # torch.Size([B, 4])
        outputs = self.activation(outputs)  # torch.Size([B, 4])
        outputs = self.dropout(outputs)  # torch.Size([B, 128])

        outputs = self.predict(outputs)  # torch.Size([B, 4])
        outputs = self.activation(outputs)  # torch.Size([B, 4])

        return outputs
```

```
class Model(torch.nn.Module):
    def __init__(self, inputs_size, hidden_size, outputs_size):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(inputs_size, hidden_size, num_layers=4, batch_first=True, dropout=0.1,
                                  bidirectional=True)
        self.bn = torch.nn.BatchNorm1d(hidden_size * 2)
        self.fc = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.predict = torch.nn.Linear(hidden_size, outputs_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def lstm_forward(self, inputs):  # torch.Size([B, 30, 5])
        outputs, (h_n, c_n) = self.lstm(inputs)  # torch.Size([B, 30, 256])
        return outputs[:, -1, :]  # torch.Size([B, 256])

    def forward(self, inputs):  # torch.Size([B, 30, 5])
        outputs = self.lstm_forward(inputs)  # torch.Size([B, 256])
        outputs = self.bn(outputs)
        outputs = self.activation(outputs)  # torch.Size([B, 256])
        outputs = self.dropout(outputs)  # torch.Size([B, 256])

        outputs = self.fc(outputs)  # torch.Size([B, 128])
        outputs = self.activation(outputs)  # torch.Size([B, 128])
        outputs = self.dropout(outputs)  # torch.Size([B, 128])

        outputs = self.predict(outputs)  # torch.Size([B, 4])
        outputs = self.activation(outputs)  # torch.Size([B, 4])

        return outputs
```

## Results 
![img](https://github.com/Ed1sonChen/sensor-data-analytics-program/blob/master/dp2.png)
![img](https://github.com/Ed1sonChen/sensor-data-analytics-program/blob/master/sp2.png)

## Problems
When I was trying to preprocessing the dataset, I made some mistakes that lead to data leaks.

```
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

```
## Findings
I tried linear models first. The training speed of linear models is very fast, and the prediction speed is also very fast. This type of model can be applied to very large datasets and is effective for sparse data. However, the ability of linear models is limited to linear functions. They cannot understand the interaction between any two input variables and cannot fit nonlinear variables well.

LSTM improves the long-term dependency problem that exists in RNNs (Recurrent Neural Networks). LSTM performs better than time-recursive neural networks and hidden Markov models (HMMs). As a nonlinear model, LSTM can be used as a complex nonlinear unit to construct larger deep neural networks.

LSTM preserves important features using various gate functions, which can effectively alleviate the gradient vanishing or exploding problems that may occur in long sequence problems. Although this phenomenon cannot be completely eliminated, LSTM performs better than linear models on longer sequence problems. With its nonlinear characteristics, LSTM can better fit multidimensional data, and matching the scale of the LSTM model and the dataset can improve data fitting.
