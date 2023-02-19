***************
* Description *
***************
A seismic sensor has been installed under a hospital bed and a family bed for monitoring vital signs of subjects on the bed. About 75 
subjects have participated in the experiment. In each experiment, a subject has lied on one bed first for a few minutes, then lie on the 
other bed for a few minutes. 

Write a sensor data analytics program to build the relationship model between the sensor data and the parameters (S, D), and 
predict S and D from future sensor data. Python language is expected for the sensor data analytics program.
The goal is to achieve a low MAE (less than 3 ideally) for two parameters (S, D) on the test data set and show a matching 
trend comparison plot of labels and prediction results (see main_plot.py as the example]). 

In the data file, each row includes sensor data (100Hz * 10 seconds) + ID + Time + H + R + S + D, which may be visualized with the command 
"./view_data.py xxx.npy 6". Here, Time is the epoch time of the first sensor data sample and can be coverted to time string with function 
epoch_time_local() in the Appendix. H, R, S and D are heartrate, respiratory rate, systolic and diastolic blood pressure.

If you choose to perform the supervised training, please use the train data set (*_train.npy) for training, and do not include
the test data set (*_test.npy) in training, which shall be used for test only. 

****************
* Delivery *
****************
1. Please save your trained model in a file and write a prediction program that reads the saved model, takes the test data set as the input, 
outputs the prediction result (S, D), and prints their MAE and us e main_plot.py to save the plots of the label and prediction result 
(see example plots from one of our algorithms in the same folder), so that we may verify your result by calling your prediction function only, without having to run your training program.

2. The goal is to achieve MAE <=3 for the parameters (S, D) on the test data set. Write a one or more pages report to summarize your 
algorithms, results and findings. 

3. Expect your source code in a professional data science coding style and framework like https://github.com/jc-audet/WOODS.