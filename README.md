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
The first layer builds a bidirectional lstm model with 2 layers, followed by activation and deactivation functions
The second layer is a linear function, followed by an activation function and a deactivation function;
The third layer is a linear function, followed by an activation function
```sh
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
```



## Results 
![img](https://github.com/Ed1sonChen/sensor-data-analytics-program/blob/master/d_MAE.png)
![img](https://github.com/Ed1sonChen/sensor-data-analytics-program/blob/master/s_MAE.png)

## Problems
When I was trying to preprocessing the dataset, I made some mistakes that lead to data leaks.

```
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

```
## Findings
I tried linear models first. The training speed of linear models is very fast, and the prediction speed is also very fast. This type of model can be applied to very large datasets and is effective for sparse data. However, the ability of linear models is limited to linear functions. They cannot understand the interaction between any two input variables and cannot fit nonlinear variables well.

LSTM improves the long-term dependency problem that exists in RNNs (Recurrent Neural Networks). LSTM performs better than time-recursive neural networks and hidden Markov models (HMMs). As a nonlinear model, LSTM can be used as a complex nonlinear unit to construct larger deep neural networks.

LSTM preserves important features using various gate functions, which can effectively alleviate the gradient vanishing or exploding problems that may occur in long sequence problems. Although this phenomenon cannot be completely eliminated, LSTM performs better than linear models on longer sequence problems. With its nonlinear characteristics, LSTM can better fit multidimensional data, and matching the scale of the LSTM model and the dataset can improve data fitting.
