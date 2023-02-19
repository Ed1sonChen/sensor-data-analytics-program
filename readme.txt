version https://git-lfs.github.com/spec/v1
oid sha256:341f39798549601f338401729cf3368b0760cd44ed0b639e935a78d002b50066
size 2219

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

### Train a model
Train a model using one objective on one dataset with one test environment. For data that requires a download, you need to provide the path to the data directory with `--data_path`.

### Dependencies
The following dependencies are required to run this project:
* numpy
* pandas
* matplotlib
* sklearn
