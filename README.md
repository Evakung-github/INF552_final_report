# INF552_project

## Introduction

Wearable devices enable us to track human behaviors continuously. Recognizing the connection between continuous sensor data and human actions may help the physicians better understand the conditions of patients. In this report, the Hidden Markov Model is utilized to extract the patterns of continuous accelerometer sequences and the extracted features are proved to be helpful in predicting Parkinson’s Disease patients’ medical states. I directly use the raw data provided and the accuracy rates are around 60% for predicting medication states and classifying severity of tremor and dyskinesia. Additional analysis and data preprocessing on the human sensory data may further recognize the human behavior patterns and increase the classification accuracy. Therefore, it may become feasible for doctors to use sensor data recorded by the portable devices as a support to prescribe the medicine.

## Model

As we were provided with the continuous collection of sensor data from wearable devices, there are more than hundred thousand timestamps in each individual file. Instead of training machine/deep learning on the dataset directly, it may be more practical and efficient to analyze data after extracting some patterns of data. 
I utilized the Hidden Markov Model (HMM) to help discover the potential patterns of human behaviors. Then, embedded the HMM before running classification and regression model on the data.

